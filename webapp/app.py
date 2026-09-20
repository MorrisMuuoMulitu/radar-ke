"""RADAR web app (Streamlit).

Upload an abdominal CT (.nii/.nii.gz or a .zip of DICOM), run the RADAR
vision-language model, and browse ranked findings with a slice viewer.

Run from the repo root:
    streamlit run webapp/app.py
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'RADAR_inference'))

os.environ.setdefault('MODEL_ROOT', os.path.join(REPO_ROOT, 'ckpt'))
os.environ.setdefault('CONFIGS_ROOT', os.path.join(REPO_ROOT, 'ckpt'))
os.environ.setdefault('HF_HOME', os.path.join(tempfile.gettempdir(), 'radar_hf'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch  # noqa: E402
from inference_service import get_model, run_case  # noqa: E402

st.set_page_config(page_title='RADAR — Abdominal CT Findings', layout='wide')

ORGAN_INDEX = {
    1: 'Adrenal gland', 2: 'Aorta', 3: 'Erector spinae', 4: 'Brain',
    5: 'Clavicle', 6: 'Large bowel', 7: 'Duodenum', 8: 'Esophagus',
    9: 'Face', 10: 'Femur', 11: 'Gallbladder', 12: 'Gluteus',
    13: 'Heart', 14: 'Hip joint', 15: 'Humerus', 16: 'Iliac artery',
    17: 'Iliac vena', 18: 'Iliopsoas', 19: 'Inferior vena cava',
    20: 'Kidney', 21: 'Liver', 22: 'Lung', 23: 'Pancreas',
    24: 'Portal vein', 25: 'Pulmonary artery', 26: 'Rib', 27: 'Sacrum',
    28: 'Scapula', 29: 'Small bowel', 30: 'Spleen', 31: 'Stomach',
    32: 'Trachea', 33: 'Bladder', 34: 'Cervical vertebrae',
    35: 'Lumbar vertebrae', 36: 'Thoracic vertebrae',
}


@st.cache_resource(show_spinner=False)
def load_model_cached():
    return get_model()


def pick_roi_size():
    """Paper default on big GPUs, smaller windows on <=10GB cards."""
    if os.environ.get('ROI_SIZE'):
        return os.environ['ROI_SIZE']
    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_gb < 10:
            os.environ['ROI_SIZE'] = '64,192,288'
            return '64,192,288 (low-VRAM mode)'
    except Exception:
        pass
    return '96,256,384 (paper default)'


def convert_dicom_zip(zip_path, work_dir):
    """Unzip DICOMs and convert to a single .nii.gz via dcm2niix."""
    dcm_dir = os.path.join(work_dir, 'dicom')
    os.makedirs(dcm_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dcm_dir)
    out_dir = os.path.join(work_dir, 'nifti')
    os.makedirs(out_dir, exist_ok=True)
    proc = subprocess.run(
        ['dcm2niix', '-o', out_dir, '-f', 'scan', '-z', 'y', dcm_dir],
        capture_output=True, text=True, timeout=600,
    )
    produced = sorted(f for f in os.listdir(out_dir) if f.endswith('.nii.gz'))
    if len(produced) != 1:
        raise RuntimeError(
            f'dcm2niix produced {len(produced)} volumes (expected exactly 1). '
            f'Stdout: {proc.stdout[-2000:]} Stderr: {proc.stderr[-2000:]}'
        )
    return os.path.join(out_dir, produced[0])


def render_slice(image, mask, axis, index, show_overlay, opacity):
    sl_img = np.take(image, index, axis=axis)
    sl_mask = np.take(mask, index, axis=axis)
    if axis == 0:
        sl_img, sl_mask = sl_img.T, sl_mask.T
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(sl_img, cmap='gray', aspect='auto')
    if show_overlay:
        masked = np.ma.masked_where(sl_mask == 0, sl_mask)
        ax.imshow(masked, cmap='tab20', alpha=opacity, vmin=0, vmax=36,
                  aspect='auto', interpolation='nearest')
    ax.axis('off')
    fig.tight_layout(pad=0)
    return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.title('Settings')
threshold = st.sidebar.slider('Score threshold', 0.0, 1.0, 0.5, 0.05)
search = st.sidebar.text_input('Filter findings (text)', '')
show_overlay = st.sidebar.checkbox('Organ mask overlay', value=True)
opacity = st.sidebar.slider('Overlay opacity', 0.1, 1.0, 0.5, 0.05)

st.title('RADAR — Abdominal CT Findings')
st.warning(
    '**Research use only — not a medical device and not a diagnosis.** '
    'Output must be reviewed by a qualified radiologist. Model covers 146 '
    'fixed findings and was trained on contrast-enhanced abdominal CT.'
)
st.caption('Repo: CC BY-NC-SA 4.0 (non-commercial). Uploaded scans stay on this '
           'machine and are deleted when you upload a new case or clear the session.')

if not torch.cuda.is_available():
    st.error('No CUDA GPU detected. RADAR inference requires a GPU.')
    st.stop()

with st.spinner('Loading RADAR model (once, ~1–2 min)…'):
    try:
        load_model_cached()
    except Exception as e:
        st.error(f'Model failed to load: {e}')
        st.stop()
st.sidebar.success(f"Model ready · window {pick_roi_size()}")

# ------------------------------------------------------------------- upload
uploaded = st.file_uploader(
    'Upload a scan: contrast-enhanced abdominal CT as .nii/.nii.gz, or a .zip of DICOM',
    type=['nii', 'gz', 'zip'],
)
if st.sidebar.button('Delete my data'):
    for key in ('result', 'work_dir'):
        st.session_state.pop(key, None)
    st.sidebar.success('Session data cleared.')

if uploaded is not None and st.button('Run inference', type='primary'):
    work_dir = tempfile.mkdtemp(prefix='radar_web_')
    old = st.session_state.pop('work_dir', None)
    if old and os.path.isdir(old):
        shutil.rmtree(old, ignore_errors=True)
    st.session_state['work_dir'] = work_dir
    up_path = os.path.join(work_dir, uploaded.name)
    with open(up_path, 'wb') as f:
        f.write(uploaded.getbuffer())
    try:
        with st.status('Processing…', expanded=True) as status:
            if up_path.endswith('.zip'):
                if shutil.which('dcm2niix') is None:
                    raise RuntimeError('dcm2niix is not installed; cannot convert DICOM.')
                st.write('Converting DICOM → NIfTI…')
                nifti_path = convert_dicom_zip(up_path, work_dir)
            elif up_path.endswith(('.nii.gz', '.nii')):
                nifti_path = up_path
            else:
                raise RuntimeError('Unsupported file type. Upload .nii/.nii.gz or a DICOM .zip.')
            st.write(f'Running RADAR on {os.path.basename(nifti_path)}…')
            result = run_case(nifti_path, os.path.join(work_dir, 'case'))
            st.session_state['result'] = result
            status.update(label='Done', state='complete')
    except Exception as e:
        st.error(f'Failed: {e}')

# ------------------------------------------------------------------ results
result = st.session_state.get('result')
if result is None:
    st.info('Upload a scan and press **Run inference**.')
    st.stop()

scores = {k: v for k, v in result['scores'].items()}
rows = [{'finding': k, 'score': (v if v is not None else float('nan'))}
        for k, v in scores.items()]
df = pd.DataFrame(rows).sort_values('score', ascending=False)
if search:
    df = df[df['finding'].str.contains(search, case=False, na=False)]
df_hi = df[df['score'] >= threshold]

c1, c2, c3 = st.columns(3)
c1.metric('Findings scored', len(df))
c2.metric(f'≥ {threshold:.2f}', int(df_hi.shape[0]))
top = df.iloc[0]
c3.metric('Top finding', f"{top['score']:.3f}", top['finding'][:40])

st.subheader(f'Ranked findings — {result["file_name"]}')
st.dataframe(
    df.assign(score=df['score'].map(lambda x: f'{x:.4f}' if pd.notna(x) else '—')),
    use_container_width=True, hide_index=True,
)
csv_buf = io.StringIO()
pd.DataFrame([{'file_name': result['file_name'], **scores}]).to_csv(
    csv_buf, index=False, encoding='utf-8-sig')
st.download_button('Download CSV', csv_buf.getvalue(),
                   file_name=f"RADAR_{result['file_name']}.csv", mime='text/csv')

# -------------------------------------------------------------------- viewer
st.subheader('Slice viewer (model view — resampled, not diagnostic quality)')
data = np.load(result['case_path'])
image = np.squeeze(data['image']).astype(np.float32)
mask = np.squeeze(data['mask']).astype(np.uint8)
present = [int(i) for i in np.unique(mask) if i != 0]
st.caption('Organs segmented in this scan: ' +
           (', '.join(f'{i} ({ORGAN_INDEX.get(i, "?")})' for i in present)
            if present else 'none'))

axis_names = {'Axial (D)': 0, 'Coronal (H)': 1, 'Sagittal (W)': 2}
axis_label = st.radio('Plane', list(axis_names), horizontal=True)
axis = axis_names[axis_label]
idx = st.slider('Slice', 0, image.shape[axis] - 1, image.shape[axis] // 2)
fig = render_slice(image, mask, axis, idx, show_overlay, opacity)
st.pyplot(fig, use_container_width=True)
plt.close(fig)
