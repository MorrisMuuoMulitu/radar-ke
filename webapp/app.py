"""RADAR research review workspace. Run: streamlit run webapp/app.py"""
import io
import os
import sys
import json
import time
import tempfile
import subprocess
import zipfile
from pathlib import Path
from html import escape
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib import colormaps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'webapp'))
sys.path.insert(0, str(REPO_ROOT / 'RADAR_inference'))
from review import (
    build_case_record,
    clear_case,
    filter_findings,
    list_case_records,
    load_case_record,
    save_case_record,
    score_table,
    structured_report,
)
os.environ.setdefault('MODEL_ROOT', str(REPO_ROOT / 'ckpt'))
os.environ.setdefault('CONFIGS_ROOT', str(REPO_ROOT / 'ckpt'))
os.environ.setdefault('HF_HOME', str(Path(tempfile.gettempdir()) / 'radar_hf'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

st.set_page_config(page_title='RADAR | CT review workspace', page_icon='◉', layout='wide')
st.markdown((REPO_ROOT / 'webapp/style.css').read_text(), unsafe_allow_html=True)
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
def hardware():
    import torch
    if not torch.cuda.is_available():
        return ('CPU', 0.0)
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 10e9:
        os.environ.setdefault('ROI_SIZE', '64,192,288')
    return props.name, props.total_memory / 1e9


@st.cache_resource(show_spinner=False)
def inference_resources():
    import threading
    from inference_service import get_model
    return get_model(), threading.Lock()


@st.cache_data(show_spinner=False, max_entries=2)
def read_example():
    import nibabel as nib
    source = REPO_ROOT / 'data/demo_cases/AC423ccbe.nii.gz'
    image = np.asarray(nib.load(str(source)).dataobj, dtype=np.float32)
    image = np.transpose(image, (2, 1, 0))
    image = np.clip(image, -300, 400)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    csv = REPO_ROOT / 'results/RADAR_infer_results_demo_8gb.csv'
    if not csv.exists():
        csv = REPO_ROOT / 'results/RADAR_infer_results_demo.csv'
    row = pd.read_csv(csv).iloc[0]
    if str(row['file_name']) != source.name:
        raise ValueError('Example scores do not match the bundled volume.')
    return {'file_name': source.name, 'scores': row.drop('file_name').to_dict(),
            'image': image, 'mask': None, 'example': True}


def case_arrays(result):
    if result.get('example'):
        return result['image'], None
    # Keep private volumes in session memory only, never in a shared data cache.
    if '_image' not in result:
        with np.load(result['case_path'], allow_pickle=False) as data:
            result['_image'] = np.squeeze(data['image']).astype(np.float32)
            result['_mask'] = np.squeeze(data['mask']).astype(np.uint8)
    return result['_image'], result['_mask']


def convert_dicom(path, work):
    import shutil
    if not shutil.which('dcm2niix'):
        raise RuntimeError('DICOM conversion requires dcm2niix. Convert to NIfTI or install dcm2niix.')
    target = work / 'dicom'
    target.mkdir()
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 8 * 1024**3:
            raise ValueError('The uncompressed archive exceeds the 8 GB limit.')
        for member in archive.infolist():
            if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError('The archive contains an invalid file path.')
        archive.extractall(target)
    output = work / 'converted'
    output.mkdir()
    proc = subprocess.run(['dcm2niix', '-o', str(output), '-f', 'scan', '-z', 'y', str(target)],
                          capture_output=True, text=True, timeout=600)
    files = list(output.glob('*.nii.gz'))
    if proc.returncode or len(files) != 1:
        raise RuntimeError(f'Conversion produced {len(files)} volumes. Upload one CT series as NIfTI. {proc.stderr[-300:]}')
    return files[0]


def start_case(upload):
    clear_case(st.session_state)
    work = Path(tempfile.mkdtemp(prefix='radar_web_'))
    st.session_state.work_dir = str(work)
    path = work / Path(upload.name).name
    path.write_bytes(upload.getbuffer())
    if not path.name.lower().endswith(('.nii', '.nii.gz', '.zip')):
        raise ValueError('Choose a .nii, .nii.gz, or DICOM .zip file.')
    started = time.monotonic()
    with st.status('Preparing your scan…', expanded=True) as status:
        if path.suffix.lower() == '.zip':
            st.write('Converting the DICOM series…')
            path = convert_dicom(path, work)
        st.write('Loading the model…' + (' Runs on CPU (no GPU) — analysis will be slow.' if hardware()[0] == 'CPU' else ' Waiting for the GPU…'))
        _, lock = inference_resources()
        from inference_service import run_case
        with lock:
            st.write('Segmenting anatomy and scoring findings. This can take several minutes.')
            if hardware()[0] == 'CPU':
                st.warning('CPU mode: a full analysis of a large scan can take tens of minutes to hours. Keep this page open.')
            result = run_case(str(path), str(work / 'case'))
        result['elapsed'] = time.monotonic() - started
        st.session_state.result = result
        status.update(label='Scan ready for review', state='complete', expanded=False)


def slice_rgb(image, mask, axis, index, brightness, contrast, overlay, opacity, selected):
    gray = np.take(image, index, axis=axis)
    gray = np.clip((gray - 0.5) * contrast + 0.5 + brightness, 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if overlay and mask is not None:
        labels = np.take(mask, index, axis=axis)
        keep = labels > 0 if selected == 0 else labels == selected
        colors = colormaps['turbo'](labels / 36)[..., :3]
        rgb[keep] = rgb[keep] * (1 - opacity) + colors[keep] * opacity
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def show_viewer(result):
    image, mask = case_arrays(result)
    with st.container(border=True):
        st.subheader('Volume explorer')
        st.caption('Orientation is not validated for diagnostic use. Display adjustments are not HU windows.')
        a, b, c = st.columns([1.1, 1, 1])
        mode = a.radio('Layout', ['Three planes', 'Single plane'], horizontal=True)
        brightness = b.slider('Brightness', -0.4, 0.4, 0.0, 0.05)
        contrast = c.slider('Contrast', 0.5, 3.0, 1.0, 0.1)
        present = [] if mask is None else [int(i) for i in np.unique(mask) if i > 0]
        selected = 0
        overlay = False
        opacity = 0.45
        if mask is not None:
            a, b, c = st.columns([1.3, 1, 1])
            selected = a.selectbox('Highlight anatomy', [0] + present,
                format_func=lambda x: 'All segmented anatomy' if x == 0 else ORGAN_INDEX.get(x, str(x)))
            overlay = b.checkbox('Show organ overlay', value=True)
            opacity = c.slider('Overlay opacity', 0.1, 0.9, 0.45, 0.05)
            if selected and st.button('Center on selected anatomy'):
                center = np.median(np.argwhere(mask == selected), axis=0).astype(int)
                for axis in range(3):
                    st.session_state[f'slice_{axis}'] = int(center[axis])
        else:
            st.caption('Example uses saved scores and the bundled scan. Segmentation overlays become available after running inference.')
        names = ['Axial', 'Coronal', 'Sagittal']
        axes = list(range(3)) if mode == 'Three planes' else [names.index(st.radio('Plane', names, horizontal=True))]
        columns = st.columns(len(axes))
        for col, axis in zip(columns, axes):
            with col:
                st.markdown(f'**{names[axis]}**')
                key = f'slice_{axis}'
                if key not in st.session_state or st.session_state[key] >= image.shape[axis]:
                    st.session_state[key] = image.shape[axis] // 2
                index = st.slider(f'{names[axis]} slice', 0, image.shape[axis] - 1, key=key)
                pixels = slice_rgb(image, mask, axis, index, brightness, contrast, overlay, opacity, selected)
                st.image(pixels, width='stretch')
                st.caption(f'Slice {index + 1} of {image.shape[axis]}')
                buf = io.BytesIO()
                Image.fromarray(pixels).save(buf, format='PNG')
                st.download_button('Save slice PNG', buf.getvalue(), f'{names[axis].lower()}_{index + 1}.png',
                                   'image/png', key=f'png_{axis}')
        st.caption(f'Volume dimensions: {image.shape[0]} × {image.shape[1]} × {image.shape[2]} voxels' +
                   (f' • {len(present)} segmented structures' if mask is not None else ''))


gpu = hardware()
with st.sidebar:
    st.markdown('''<div class="brand"><svg width="42" height="42" viewBox="0 0 42 42" fill="none"><circle cx="21" cy="21" r="18" stroke="#64c6cc" stroke-width="2"/><circle cx="21" cy="21" r="10" stroke="#64c6cc"/><path d="M21 3v36M3 21h36" stroke="#64c6cc"/><circle cx="21" cy="21" r="3" fill="#fff"/></svg><div><strong>RADAR</strong><small>Abdominal CT workspace</small></div></div>''', unsafe_allow_html=True)
    st.subheader('Case workspace')
    st.caption('Import a scan or explore the included example.')
    upload = st.file_uploader('Import CT scan', type=['nii', 'gz', 'zip'], help='One NIfTI volume or a ZIP containing one DICOM series.')
    run = st.button('Analyze scan', type='primary', width='stretch', disabled=upload is None)
    if st.button('Open example case', width='stretch'):
        try:
            example = read_example()
            clear_case(st.session_state)
            st.session_state.result = dict(example)
        except Exception as exc:
            st.error(f'Example unavailable: {exc}')
    history = list_case_records()
    if history:
        st.divider()
        st.subheader('Case history')
        history_labels = {
            f'{item["file_name"]} • {item["status"]} • {item["saved_at"][:10]}': item['case_id']
            for item in history
        }
        selected_history = st.selectbox('Saved cases', list(history_labels.keys()))
        if st.button('Open saved review', width='stretch'):
            record = load_case_record(history_labels[selected_history])
            clear_case(st.session_state)
            st.session_state.result = {
                'file_name': record.get('file_name', 'saved_case'),
                'scores': record.get('scores', {}),
                'example': record.get('example', False),
                'history_only': True,
                'saved_record': record,
            }
            st.session_state.review_status = record.get('status', 'Not started')
            st.session_state.review_flags = record.get('shortlist', [])
            st.session_state.review_notes = record.get('notes', '')
            st.rerun()
    st.divider()
    st.subheader('Compute')
    if gpu[0] == 'CPU':
        st.caption('No CUDA GPU detected — analysis runs on CPU and is slow.')
        st.caption('Expect tens of minutes to hours per scan. The example case loads instantly.')
    else:
        st.caption(f'{gpu[0]}\n\n{gpu[1]:.1f} GB total GPU memory')
        st.caption('Compact inference windows' if gpu[1] < 10 else 'Standard inference windows')
    if st.session_state.get('result') is not None:
        st.divider()
        if st.button('Clear case and delete uploads', width='stretch'):
            clear_case(st.session_state)
            st.rerun()
    st.caption('Uploads are processed locally. Clear the case to remove its temporary files. Closing the browser alone does not delete files.')

st.markdown('<div class="workspace-header"><div><h1>RADAR</h1><p>Abdominal CT review workspace</p></div><span class="status-chip">Research workspace</span></div>', unsafe_allow_html=True)
if run:
    try:
        start_case(upload)
    except Exception as exc:
        st.error(f'Analysis could not complete: {exc}')
        st.info('Check the scan format and GPU memory, then try again. The example remains available.')

result = st.session_state.get('result')
if result is None:
    st.markdown('<div class="welcome"><h2>A closer look at every scan.</h2><p>Bring anatomy and model findings into one review surface. Inspect three planes, focus on an organ, and capture the findings that deserve a closer look.</p></div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.subheader('Explore the volume')
        st.write('Navigate axial, coronal, and sagittal views with independent slice and display controls.')
    with b:
        st.subheader('Find what matters')
        st.write('Search 146 finding scores, filter by anatomy, and build a shortlist for review.')
    with c:
        st.subheader('Keep the evidence')
        st.write('Export scores, save individual slices, and download your notes with a structured review.')
    st.info('Choose “Open example case” in the sidebar to explore immediately, or import a CT scan to run the model.')
else:
    rows = score_table(result['scores'])
    example = result.get('example', False)
    detail = 'Bundled example • Previously computed model scores' if example else 'Local analysis • Model results ready'
    st.markdown(f'<div class="case-strip"><strong>{escape(result["file_name"])}</strong><small>{detail}</small></div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric('Scored findings', f'{rows.score.notna().sum()} / {len(rows)}')
    b.metric('Anatomical groups', rows.organ.nunique())
    c.metric('Highest model score', f'{rows.score.max():.3f}' if rows.score.notna().any() else 'Unavailable')
    d.metric('Review status', st.session_state.get('review_status', 'Not started'))
    st.caption('Scores compare predefined positive and negative prompts. They are not calibrated disease probabilities.')
    viewer, findings, review_tab = st.tabs(['Scan explorer', 'Findings', 'Review & export'])
    with viewer:
        if result.get('history_only'):
            st.info('This saved review contains notes, scores, and shortlist metadata. Reopen or reanalyze the CT scan to view image slices.')
        else:
            show_viewer(result)
    with findings:
        st.subheader('Finding explorer')
        a, b = st.columns([1.4, 1])
        query = a.text_input('Search findings', placeholder='Try liver, cyst, or calcification…')
        organs = b.multiselect('Filter by anatomy', sorted(rows.organ.unique()))
        a, b, c = st.columns([1, 1, 1])
        threshold = a.slider('Model score threshold', 0.0, 1.0, 0.5, 0.05)
        above = b.checkbox('Only show scores above threshold')
        order = c.selectbox('Sort by', ['Highest score', 'Lowest score', 'Anatomy'])
        filtered = filter_findings(rows, query, organs, threshold, above)
        if order == 'Lowest score':
            filtered = filtered.sort_values('score', na_position='last')
        elif order == 'Anatomy':
            filtered = filtered.sort_values(['organ', 'finding'])
        st.caption(f'{len(filtered)} matching findings • {int((rows.score >= threshold).sum())} total at or above {threshold:.2f} • Missing predictions remain blank')
        if filtered.empty:
            st.info('No findings match these filters. Try a different term, remove an anatomy filter, or lower the threshold.')
        else:
            st.dataframe(filtered[['organ', 'finding', 'score']], hide_index=True, width='stretch',
                column_config={'organ': 'Anatomy', 'finding': 'Finding',
                               'score': st.column_config.ProgressColumn('Model score', min_value=0, max_value=1, format='%.3f')}, height=520)
        st.download_button('Export filtered findings', filtered.drop(columns='key').to_csv(index=False).encode('utf-8-sig'), 'radar_filtered_findings.csv', 'text/csv')
    with review_tab:
        st.subheader('Review notebook')
        st.caption('Your selections and notes stay with this case for the current session. Download them before clearing the case.')
        labels = dict(zip(rows.key, rows.organ + ' / ' + rows.finding))
        st.multiselect('Shortlist findings for follow-up', rows.key.tolist(), format_func=lambda x: labels.get(x, x), key='review_flags')
        st.selectbox('Review status', ['Not started', 'In progress', 'Reviewed'], key='review_status')
        st.text_area('Reviewer notes', placeholder='Record observations, questions, and follow-up considerations…', height=180, key='review_notes')
        export = build_case_record(
            result,
            st.session_state.review_status,
            st.session_state.review_flags,
            st.session_state.review_notes,
            saved_at=datetime.now(timezone.utc).isoformat(),
        )
        report = structured_report(export)
        st.subheader('Structured report draft')
        st.text_area('Draft report', value=report, height=260)
        if st.button('Save case to history', width='stretch'):
            saved = save_case_record(export)
            st.success(f'Saved review for {saved["file_name"]}.')
            st.rerun()
        a, b, c = st.columns(3)
        a.download_button('Download review JSON', json.dumps(export, indent=2, ensure_ascii=False), 'radar_review.json', 'application/json', width='stretch')
        b.download_button('Download report TXT', report, 'radar_structured_report.txt', 'text/plain', width='stretch')
        c.download_button('Download all finding scores', rows.drop(columns='key').to_csv(index=False).encode('utf-8-sig'), 'radar_all_findings.csv', 'text/csv', width='stretch')

st.markdown('<div class="research-note">Research use only. Not a medical device or a diagnosis. Outputs require qualified review. Trained for contrast-enhanced abdominal CT.</div>', unsafe_allow_html=True)
