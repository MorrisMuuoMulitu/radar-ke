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
    WINDOW_PRESETS,
    apply_window,
    build_case_record,
    clear_case,
    delete_case_record,
    filter_worklist,
    FINDING_STATES,
    filter_findings,
    list_case_records,
    load_case_record,
    REVIEW_STATUSES,
    save_case_record,
    score_table,
    structured_report,
    worklist_rows,
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
ORGAN_NAME_TO_INDEX = {name: index for index, name in ORGAN_INDEX.items()}


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
    csv = REPO_ROOT / 'results/RADAR_infer_results_demo_8gb.csv'
    if not csv.exists():
        csv = REPO_ROOT / 'results/RADAR_infer_results_demo.csv'
    row = pd.read_csv(csv).iloc[0]
    if str(row['file_name']) != source.name:
        raise ValueError('Example scores do not match the bundled volume.')
    return {'file_name': source.name, 'scores': row.drop('file_name').to_dict(),
            'image': image, 'mask': None, 'example': True, 'display_mode': 'hu'}


def case_arrays(result):
    if result.get('example'):
        return result['image'], None, result.get('display_mode', 'normalized')
    # Keep private volumes in session memory only, never in a shared data cache.
    if '_image' not in result:
        with np.load(result['case_path'], allow_pickle=False) as data:
            source = data['display_hu'] if 'display_hu' in data.files else data['image']
            result['_image'] = np.squeeze(source).astype(np.float32)
            result['_mask'] = np.squeeze(data['mask']).astype(np.uint8)
            result['_display_mode'] = 'hu' if 'display_hu' in data.files else 'normalized'
    return result['_image'], result['_mask'], result.get('_display_mode', 'normalized')


def mask_center(mask, label):
    if mask is None or not label:
        return None
    points = np.argwhere(mask == label)
    if points.size == 0:
        return None
    return np.median(points, axis=0).astype(int)


def center_viewer_on_label(mask, label):
    center = mask_center(mask, label)
    if center is None:
        return False
    for axis in range(3):
        st.session_state[f'slice_{axis}'] = int(center[axis])
    return True


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
        st.session_state['_nav_view'] = 'Review workspace'
        status.update(label='Scan ready for review', state='complete', expanded=False)
        st.rerun()


def open_saved_case(case_id):
    """Load a saved review record into the session and switch to the workspace."""
    record = load_case_record(case_id)
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
    st.session_state.finding_states = record.get('finding_states', {})
    st.session_state.review_notes = record.get('notes', '')
    st.session_state.review_context = record.get('clinical_context', '')
    st.session_state['_nav_view'] = 'Review workspace'


def slice_rgb(image, mask, axis, index, display_mode, window_level, window_width, overlay, opacity, selected):
    gray = np.take(image, index, axis=axis)
    if display_mode == 'hu':
        gray = apply_window(gray, window_level, window_width)
    else:
        gray = np.clip(gray, 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if overlay and mask is not None:
        labels = np.take(mask, index, axis=axis)
        keep = labels > 0 if selected == 0 else labels == selected
        colors = colormaps['turbo'](labels / 36)[..., :3]
        rgb[keep] = rgb[keep] * (1 - opacity) + colors[keep] * opacity
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def show_viewer(result):
    image, mask, display_mode = case_arrays(result)
    pending_jump = st.session_state.pop('pending_anatomy_jump', None)
    if pending_jump and center_viewer_on_label(mask, pending_jump):
        st.session_state.highlight_anatomy = int(pending_jump)
    with st.container(border=True):
        st.subheader('Volume explorer')
        st.caption('Orientation is not validated for diagnostic use. Window presets use HU when available.')
        a, b, c = st.columns([1.1, 1, 1])
        mode = a.radio('Layout', ['Three planes', 'Single plane'], horizontal=True)
        preset_name = b.selectbox('Window preset', list(WINDOW_PRESETS.keys()))
        preset = WINDOW_PRESETS[preset_name]
        window_level = c.number_input('Window level', value=int(preset['center']), step=10)
        window_width = c.number_input('Window width', min_value=1, value=int(preset['width']), step=10)
        if display_mode != 'hu':
            st.caption('This case uses normalized display data from an older model output. Reanalyze the scan to enable true HU windowing.')
        present = [] if mask is None else [int(i) for i in np.unique(mask) if i > 0]
        selected = 0
        overlay = False
        opacity = 0.45
        if mask is not None:
            a, b, c = st.columns([1.3, 1, 1])
            if st.session_state.get('highlight_anatomy') not in [0] + present:
                st.session_state.highlight_anatomy = 0
            selected = a.selectbox('Highlight anatomy', [0] + present,
                format_func=lambda x: 'All segmented anatomy' if x == 0 else ORGAN_INDEX.get(x, str(x)),
                key='highlight_anatomy')
            overlay = b.checkbox('Show organ overlay', value=True)
            opacity = c.slider('Overlay opacity', 0.1, 0.9, 0.45, 0.05)
            if selected and st.button('Center on selected anatomy'):
                center_viewer_on_label(mask, selected)
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
                nav_prev, nav_next = st.columns(2)
                if nav_prev.button(f'Previous {names[axis].lower()} slice', key=f'prev_{axis}', width='stretch'):
                    st.session_state[key] = max(0, st.session_state[key] - 1)
                    st.rerun()
                if nav_next.button(f'Next {names[axis].lower()} slice', key=f'next_{axis}', width='stretch'):
                    st.session_state[key] = min(image.shape[axis] - 1, st.session_state[key] + 1)
                    st.rerun()
                index = st.slider(f'{names[axis]} slice', 0, image.shape[axis] - 1, key=key)
                pixels = slice_rgb(image, mask, axis, index, display_mode, window_level, window_width, overlay, opacity, selected)
                st.image(pixels, width='stretch')
                st.caption(f'Slice {index + 1} of {image.shape[axis]}')
                buf = io.BytesIO()
                Image.fromarray(pixels).save(buf, format='PNG')
                st.download_button('Save slice PNG', buf.getvalue(), f'{names[axis].lower()}_{index + 1}.png',
                                   'image/png', key=f'png_{axis}')
        st.caption(f'Volume dimensions: {image.shape[0]} × {image.shape[1]} × {image.shape[2]} voxels' +
                   (f' • {len(present)} segmented structures' if mask is not None else ''))


def show_worklist():
    st.markdown('<div class="workspace-header"><div><h1>Case worklist</h1><p>Saved reviews across the workspace</p></div><span class="status-chip">Worklist</span></div>', unsafe_allow_html=True)
    rows = worklist_rows()
    if rows.empty:
        st.markdown('<div class="welcome"><h2>No saved cases yet.</h2><p>Analyze a scan and use “Save case to history” in the Review &amp; export tab — saved cases appear here with their review status and likely-present findings.</p></div>', unsafe_allow_html=True)
        return
    a, b, c, d = st.columns(4)
    a.metric('Total cases', len(rows))
    b.metric('Not started', int((rows['status'] == 'Not started').sum()))
    c.metric('In progress', int((rows['status'] == 'In progress').sum()))
    d.metric('Reviewed', int((rows['status'] == 'Reviewed').sum()))
    st.divider()
    a, b, c = st.columns([1.4, 1, 1])
    query = a.text_input('Search cases', placeholder='Filename…')
    statuses = b.multiselect('Filter by status', REVIEW_STATUSES)
    sort = c.selectbox('Sort by', ['Newest', 'Oldest', 'Filename', 'Status'])
    filtered = filter_worklist(rows, query, statuses, sort)
    st.caption(f'{len(filtered)} of {len(rows)} cases')
    if filtered.empty:
        st.info('No cases match these filters. Try a different filename or clear the status filter.')
    for _, row in filtered.iterrows():
        with st.container(border=True):
            col_a, col_b, col_c, col_d = st.columns([1.7, 1, 1.4, 1])
            with col_a:
                st.markdown(f'**{escape(row["file_name"])}**')
                saved_text = str(row['saved_at'])[:19].replace('T', ' ') if row.get('saved_at') else '—'
                st.caption(f'Saved {saved_text} • ID {row["case_id"][:8]}')
            with col_b:
                st.markdown(f'**{escape(row["status"])}**')
                st.caption(f'Shortlist: {row["shortlist_count"]}')
            with col_c:
                if row['likely_present_count']:
                    st.markdown(f'**{int(row["likely_present_count"])} likely present**')
                    preview = str(row['likely_present'])
                    st.caption(preview[:90] + ('…' if len(preview) > 90 else ''))
                else:
                    st.caption('No likely-present findings marked')
            with col_d:
                if st.button('Open case', key=f'open_{row["case_id"]}', width='stretch'):
                    open_saved_case(row['case_id'])
                    st.rerun()
                report = structured_report(load_case_record(row['case_id']))
                st.download_button('Report TXT', report, f'{Path(row["file_name"]).stem}_report.txt',
                                   'text/plain', key=f'report_{row["case_id"]}', width='stretch')
                if st.button('Delete', key=f'del_{row["case_id"]}', width='stretch'):
                    delete_case_record(row['case_id'])
                    st.rerun()
    st.download_button('Download worklist CSV',
                       filtered.drop(columns='case_id').to_csv(index=False).encode('utf-8-sig'),
                       'radar_worklist.csv', 'text/csv')


# Consume view-switch intents BEFORE the View radio widget is instantiated
# (widget keys cannot be modified after the widget is created).
_nav_intent = st.session_state.pop('_nav_view', None)
if _nav_intent:
    st.session_state.app_view = _nav_intent

gpu = hardware()
with st.sidebar:
    st.markdown('''<div class="brand"><svg width="42" height="42" viewBox="0 0 42 42" fill="none"><circle cx="21" cy="21" r="18" stroke="#64c6cc" stroke-width="2"/><circle cx="21" cy="21" r="10" stroke="#64c6cc"/><path d="M21 3v36M3 21h36" stroke="#64c6cc"/><circle cx="21" cy="21" r="3" fill="#fff"/></svg><div><strong>RADAR</strong><small>Abdominal CT workspace</small></div></div>''', unsafe_allow_html=True)
    st.radio('View', ['Review workspace', 'Case worklist'], horizontal=True, key='app_view')
    st.subheader('Case workspace')
    st.caption('Import a scan or explore the included example.')
    upload = st.file_uploader('Import CT scan', type=['nii', 'gz', 'zip'], help='One NIfTI volume or a ZIP containing one DICOM series.')
    run = st.button('Analyze scan', type='primary', width='stretch', disabled=upload is None)
    if st.button('Open example case', width='stretch'):
        try:
            example = read_example()
            clear_case(st.session_state)
            st.session_state.result = dict(example)
            st.session_state['_nav_view'] = 'Review workspace'
            st.rerun()
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
            open_saved_case(history_labels[selected_history])
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
if st.session_state.get('app_view', 'Review workspace') == 'Case worklist':
    show_worklist()
elif result is None:
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
    st.session_state.setdefault('finding_states', {})
    a, b, c, d = st.columns(4)
    a.metric('Scored findings', f'{rows.score.notna().sum()} / {len(rows)}')
    b.metric('Anatomical groups', rows.organ.nunique())
    c.metric('Likely present', sum(1 for state in st.session_state.finding_states.values() if state == 'Likely present'))
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
        row_lookup = rows.set_index('key').to_dict('index')
        st.multiselect('Shortlist findings for follow-up', rows.key.tolist(), format_func=lambda x: labels.get(x, x), key='review_flags')
        st.markdown('**Finding review states**')
        state_a, state_b = st.columns([1.6, 1])
        if st.session_state.get('selected_finding') not in rows.key.tolist():
            st.session_state.selected_finding = rows.key.iloc[0]
        chosen_finding = state_a.selectbox('Finding to mark', rows.key.tolist(),
            format_func=lambda x: labels.get(x, x), key='selected_finding')
        current_state = st.session_state.finding_states.get(chosen_finding, 'Needs review')
        chosen_state = state_b.selectbox('Finding state', FINDING_STATES, index=FINDING_STATES.index(current_state))
        selected_row = row_lookup.get(chosen_finding, {})
        selected_organ = selected_row.get('organ')
        organ_label = ORGAN_NAME_TO_INDEX.get(selected_organ)
        jump_available = False
        if organ_label and not result.get('history_only'):
            _, review_mask, _ = case_arrays(result)
            jump_available = mask_center(review_mask, organ_label) is not None
        if st.button('Jump to anatomy in viewer', width='stretch', disabled=not jump_available):
            st.session_state.pending_anatomy_jump = organ_label
            st.toast(f'Opening {selected_organ} in the scan explorer.')
            st.rerun()
        if organ_label and not jump_available:
            st.caption(f'Anatomy jump will be available when {selected_organ} segmentation exists for this case.')
        if st.button('Apply finding state', width='stretch'):
            st.session_state.finding_states[chosen_finding] = chosen_state
            st.rerun()
        if st.session_state.finding_states:
            state_rows = [
                {'finding': labels.get(key, key), 'state': state}
                for key, state in st.session_state.finding_states.items()
            ]
            st.dataframe(pd.DataFrame(state_rows), hide_index=True, width='stretch', height=180)
        st.selectbox('Review status', REVIEW_STATUSES, key='review_status')
        st.text_input('Clinical context / indication', placeholder='e.g. RUQ pain, known hepatic lesion, follow-up', key='review_context')
        st.text_area('Reviewer notes', placeholder='Record observations, questions, and follow-up considerations…', height=180, key='review_notes')
        export = build_case_record(
            result,
            st.session_state.review_status,
            st.session_state.review_flags,
            st.session_state.review_notes,
            finding_states=st.session_state.finding_states,
            saved_at=datetime.now(timezone.utc).isoformat(),
            clinical_context=st.session_state.get('review_context', ''),
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
