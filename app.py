import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from utils.model_loader import (
    list_model_candidates,
    list_notebook_candidates,
    load_emotion_model,
)
from utils.preprocessing import (
    EMOTION_LABELS,
    EMOTION_LABELS_ID,
    EMOTION_SUGGESTIONS,
    detect_largest_face,
    draw_face_box,
    image_file_to_rgb,
    predict_emotion,
    prepare_face_for_model,
)


st.set_page_config(page_title="Deteksi Emosi Wajah", layout="wide")


CUSTOM_CSS = """
<style>
    :root {
        --blue-950: #061b42;
        --blue-900: #0b2a66;
        --blue-800: #174ea6;
        --blue-700: #2563eb;
        --blue-600: #3b82f6;
        --blue-100: #dbeafe;
        --blue-050: #f5f9ff;
        --white: #ffffff;
        --ink: #0b1f44;
        --muted: #667085;
        --line: #e6edf7;
        --soft: #f8fbff;
        --shadow: 0 16px 36px rgba(23, 78, 166, 0.15);
    }

    .stApp {
        background:
            radial-gradient(circle at 86% 8%, rgba(96, 165, 250, 0.24), transparent 24%),
            linear-gradient(180deg, #eaf3ff 0%, #f8fbff 38%, #ffffff 100%);
        color: var(--ink);
    }

    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }

    [data-testid="stToolbar"],
    #MainMenu,
    footer {
        visibility: hidden;
        height: 0;
    }

    .main .block-container {
        max-width: 1180px;
        padding: 4.4rem 2.2rem 2.4rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef6ff 0%, #ffffff 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 3.25rem;
    }

    .side-hero {
        padding: 1rem;
        border-radius: 18px;
        background: linear-gradient(145deg, var(--blue-900) 0%, var(--blue-700) 58%, #60a5fa 100%);
        color: var(--white);
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .hero + div[data-testid="stHorizontalBlock"] {
        margin-top: 1rem;
        margin-bottom: 1.35rem;
    }

    .home-action-gap {
        height: 1rem;
    }

    .side-hero p {
        margin: 0;
        color: rgba(255,255,255,0.9);
        line-height: 1.55;
        font-size: 0.94rem;
    }

    .side-label {
        margin: 1rem 0 0.5rem;
        font-size: 0.76rem;
        line-height: 1;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 900;
        color: #5b6b86;
    }

    [data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        min-height: 2.9rem;
        justify-content: flex-start;
        border-radius: 14px;
        border: 1px solid #d8e7fb;
        background: rgba(255,255,255,0.9);
        color: var(--ink);
        box-shadow: 0 8px 18px rgba(23, 78, 166, 0.05);
        font-weight: 800;
        padding-left: 1rem;
        transition: all 180ms ease;
    }

    [data-testid="stSidebar"] div.stButton > button:hover {
        transform: translateY(-2px);
        border-color: #93c5fd;
        background: #e8f2ff;
        color: var(--blue-700);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.12);
    }

    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        border: 0;
        background: linear-gradient(135deg, var(--blue-800), var(--blue-600));
        color: var(--white);
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.22);
    }

    .side-card {
        padding: 1rem;
        border-radius: 16px;
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        color: var(--muted);
        line-height: 1.6;
        font-size: 0.92rem;
    }

    .side-model {
        margin-top: 0.5rem;
        padding: 0.85rem;
        border-radius: 16px;
        background: #dbeafe;
        border: 1px solid #bfdbfe;
        color: var(--blue-950);
        font-size: 0.9rem;
        overflow-wrap: anywhere;
    }

    h1, h2, h3, p {
        letter-spacing: 0;
    }

    .page-header {
        padding: clamp(1.2rem, 2.4vw, 1.7rem);
        border-radius: 24px;
        background:
            radial-gradient(circle at 92% 20%, rgba(96,165,250,0.32), transparent 28%),
            linear-gradient(135deg, var(--blue-950), var(--blue-800));
        border: 1px solid rgba(255,255,255,0.58);
        box-shadow: 0 18px 42px rgba(8, 32, 74, 0.18);
        margin: 0.6rem 0 1.2rem;
        overflow: hidden;
    }

    .page-header .page-title,
    .page-header .page-subtitle {
        color: var(--white);
    }

    .page-header .page-subtitle {
        color: rgba(255,255,255,0.82);
        margin-bottom: 0;
    }

    .page-shell {
        display: grid;
        gap: 1rem;
    }

    .hero {
        min-height: 470px;
        display: grid;
        grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
        gap: clamp(1.2rem, 4vw, 3rem);
        align-items: center;
        padding: clamp(1rem, 2.8vw, 2rem);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 26px;
        background:
            radial-gradient(circle at 90% 18%, rgba(255,255,255,0.24), transparent 26%),
            linear-gradient(135deg, var(--blue-950) 0%, var(--blue-800) 56%, var(--blue-600) 100%);
        box-shadow: 0 22px 50px rgba(8, 32, 74, 0.22);
        backdrop-filter: blur(10px);
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        color: #ffffff;
        font-weight: 850;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0 0 1rem;
        color: var(--white);
        font-size: clamp(2.15rem, 5.2vw, 4.35rem);
        line-height: 1.02;
        font-weight: 900;
    }

    .hero h1 span {
        color: #bfdbfe;
    }

    .hero-copy {
        max-width: 650px;
        color: rgba(255,255,255,0.82);
        font-size: clamp(1rem, 1.6vw, 1.12rem);
        line-height: 1.75;
        margin: 0;
    }

    .hero-actions {
        display: grid;
        grid-template-columns: minmax(190px, 270px) minmax(0, 1fr);
        gap: 1rem;
        align-items: center;
        margin-top: 1.5rem;
    }

    .helper-note {
        color: var(--blue-800);
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 0.75rem 0.9rem;
        font-weight: 750;
        line-height: 1.45;
        font-size: 0.95rem;
    }

    div.stButton > button[kind="primary"] {
        min-height: 3.15rem;
        border: 0;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--blue-800), var(--blue-600));
        color: var(--white);
        box-shadow: 0 14px 30px rgba(37, 99, 235, 0.24);
        font-weight: 900;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0f3f91, #2563eb);
        color: var(--white);
        transform: translateY(-1px);
    }

    .visual-panel {
        position: relative;
        min-height: 390px;
        border-radius: 28px;
        background:
            linear-gradient(145deg, rgba(219,234,254,0.9), rgba(255,255,255,0.76));
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.65), var(--shadow);
        overflow: hidden;
    }

    .visual-panel::before {
        content: "";
        position: absolute;
        inset: 46px;
        border-radius: 32px;
        border: 3px solid rgba(37,99,235,0.16);
        background: rgba(255,255,255,0.34);
    }

    .student-avatar {
        position: absolute;
        left: 50%;
        top: 51%;
        width: min(210px, 54%);
        aspect-ratio: 1;
        transform: translate(-50%, -50%);
        border-radius: 50%;
        background:
            radial-gradient(circle at 36% 42%, var(--blue-800) 0 4px, transparent 5px),
            radial-gradient(circle at 64% 42%, var(--blue-800) 0 4px, transparent 5px),
            radial-gradient(ellipse at 50% 62%, var(--blue-800) 0 18%, transparent 19%),
            linear-gradient(180deg, #dbeafe 0%, #bfdbfe 100%);
        border: 13px solid var(--white);
        box-shadow: 0 24px 52px rgba(37, 99, 235, 0.18);
    }

    .student-avatar::before {
        content: "";
        position: absolute;
        left: 21%;
        right: 21%;
        top: -14%;
        height: 34%;
        border-radius: 999px 999px 18px 18px;
        background: var(--blue-950);
    }

    .scan-corner {
        position: absolute;
        width: 56px;
        height: 56px;
        border-color: var(--blue-600);
        opacity: 0.55;
    }

    .scan-corner.a { left: 19%; top: 18%; border-top: 5px solid; border-left: 5px solid; border-radius: 12px 0 0 0; }
    .scan-corner.b { right: 19%; top: 18%; border-top: 5px solid; border-right: 5px solid; border-radius: 0 12px 0 0; }
    .scan-corner.c { left: 19%; bottom: 18%; border-bottom: 5px solid; border-left: 5px solid; border-radius: 0 0 0 12px; }
    .scan-corner.d { right: 19%; bottom: 18%; border-bottom: 5px solid; border-right: 5px solid; border-radius: 0 0 12px 0; }

    .floating-result {
        position: absolute;
        right: 7%;
        top: 28%;
        width: 170px;
        padding: 1.05rem;
        border-radius: 24px;
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        box-shadow: 0 24px 48px rgba(37,99,235,0.16);
        text-align: center;
    }

    .floating-result strong {
        display: block;
        color: var(--blue-700);
        font-size: 1.1rem;
        margin: 0.35rem 0 0.2rem;
    }

    .floating-result .result-percent {
        display: block;
        color: var(--blue-950);
        font-size: 1.75rem;
        font-weight: 900;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1.1rem;
        padding: 1.1rem;
        border-radius: 24px;
        background: linear-gradient(135deg, var(--blue-900), var(--blue-700));
        box-shadow: var(--shadow);
    }

    .feature-card,
    .content-card,
    .mini-card,
    .result-card,
    .empty-state {
        border-radius: 20px;
        background: rgba(255,255,255,0.96);
        border: 1px solid var(--line);
        box-shadow: 0 10px 28px rgba(37,99,235,0.07);
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .mini-card {
        padding: 1.15rem;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }

    .mini-card h3 {
        margin: 0;
        color: var(--blue-950);
        font-size: 1.08rem;
        line-height: 1.3;
    }

    .mini-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .feature-card {
        padding: 1.25rem;
        min-height: 205px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        overflow: hidden;
    }

    .feature-card h3,
    .content-card h2,
    .result-card h2,
    .empty-state h2 {
        margin: 0 0 0.38rem;
        color: var(--blue-950);
        font-size: 1.1rem;
        line-height: 1.25;
    }

    .feature-card p,
    .content-card p,
    .empty-state p {
        color: var(--muted);
        line-height: 1.52;
        margin: 0;
        font-size: 0.92rem;
    }

    .feature-icon,
    .result-icon {
        width: 52px;
        height: 52px;
        border-radius: 16px;
        background: #dbeafe;
        display: grid;
        place-items: center;
        margin-bottom: 0.8rem;
        flex: 0 0 auto;
    }

    .content-card {
        padding: clamp(1.1rem, 2vw, 1.5rem);
        margin-bottom: 1rem;
    }

    .tips-card {
        min-height: 460px;
        height: 100%;
        box-sizing: border-box;
        overflow: hidden;
    }

    .tips-card .step-list {
        gap: 1rem;
    }

    .page-title {
        margin: 0 0 0.65rem;
        color: var(--blue-950);
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        line-height: 1.1;
        font-weight: 900;
    }

    .page-subtitle {
        max-width: 780px;
        margin: 0 0 1rem;
        color: var(--muted);
        line-height: 1.7;
        font-size: 1rem;
    }

    .camera-grid {
        display: grid;
        grid-template-columns: minmax(300px, 0.85fr) minmax(340px, 1fr);
        gap: 1rem;
        align-items: stretch;
        margin-top: 1rem;
    }

    [data-testid="stCameraInput"] {
        border-radius: 22px;
        border: 1px solid var(--line);
        background: var(--white);
        padding: 1rem;
        box-shadow: 0 14px 34px rgba(37,99,235,0.10);
        min-height: 460px;
        height: 100%;
        box-sizing: border-box;
    }

    [data-testid="stCameraInput"] > div {
        height: 100%;
    }

    [data-testid="stCameraInput"] button {
        border-radius: 14px;
        min-height: 2.75rem;
        font-weight: 850;
        width: 100%;
    }

    .media-card {
        padding: 1rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.96);
        border: 1px solid var(--line);
        box-shadow: 0 12px 30px rgba(37,99,235,0.08);
        margin: 1rem 0;
        box-sizing: border-box;
    }

    .media-card h3 {
        margin: 0 0 0.75rem;
        color: var(--blue-950);
        font-size: 1.05rem;
        line-height: 1.25;
    }

    .media-card img,
    .face-preview-img {
        width: 100%;
        height: auto;
        max-height: 390px;
        object-fit: contain;
        background: #eff6ff;
        display: block;
        border-radius: 16px;
        border: 1px solid #d8e7fb;
    }

    .face-preview {
        display: grid;
        grid-template-columns: 120px minmax(0, 1fr);
        gap: 0.9rem;
        align-items: center;
    }

    .face-preview-text {
        color: var(--muted);
        line-height: 1.5;
        font-size: 0.92rem;
    }

    .prob-card {
        padding: 1rem;
        border-radius: 22px;
        background: rgba(255,255,255,0.96);
        border: 1px solid var(--line);
        box-shadow: 0 12px 30px rgba(37,99,235,0.08);
        margin-top: 1rem;
    }

    .prob-card h3 {
        margin: 0 0 0.95rem;
        color: var(--blue-950);
        font-size: 1.05rem;
        line-height: 1.25;
    }

    .prob-list {
        display: grid;
        gap: 0.72rem;
    }

    .prob-row {
        display: grid;
        grid-template-columns: 112px minmax(0, 1fr) 58px;
        gap: 0.75rem;
        align-items: center;
    }

    .prob-label {
        color: var(--blue-950);
        font-weight: 850;
        font-size: 0.92rem;
        overflow-wrap: anywhere;
    }

    .prob-track {
        height: 14px;
        border-radius: 999px;
        background: #dbeafe;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        min-width: 3px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--blue-700), #60a5fa);
    }

    .prob-value {
        color: var(--blue-950);
        font-weight: 900;
        text-align: right;
        font-size: 0.9rem;
    }

    .prob-row.is-top {
        padding: 0.65rem;
        border-radius: 16px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
    }

    .prob-row.is-top .prob-track {
        background: #bfdbfe;
    }

    .result-card {
        padding: 1.35rem;
        margin: 1rem 0;
        min-height: 390px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .emotion-name {
        color: var(--blue-700);
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 900;
        line-height: 1;
        margin: 0.2rem 0 0.5rem;
    }

    .confidence {
        color: var(--blue-950);
        font-size: 1.25rem;
        font-weight: 900;
        margin-bottom: 0.7rem;
    }

    .suggestion {
        padding: 1rem;
        border-radius: 16px;
        background: #dbeafe;
        border: 1px solid #bfdbfe;
        color: var(--blue-950);
        line-height: 1.65;
        margin-top: 0.8rem;
        overflow-wrap: anywhere;
    }

    .warning-note,
    .disclaimer,
    .notice-card {
        padding: 1rem;
        border-radius: 16px;
        line-height: 1.6;
    }

    .warning-note {
        background: #dbeafe;
        border: 1px solid #bfdbfe;
        color: var(--blue-950);
    }

    .disclaimer {
        margin-top: 1rem;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: var(--blue-800);
    }

    .notice-card {
        margin-top: 1rem;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: var(--blue-950);
        font-weight: 750;
    }

    .empty-state {
        padding: 1.4rem;
        min-height: 240px;
        display: grid;
        align-content: center;
        justify-items: start;
    }

    .step-list {
        display: grid;
        gap: 0.85rem;
        margin-top: 0.8rem;
    }

    .step-item {
        display: grid;
        grid-template-columns: 40px minmax(0, 1fr);
        gap: 0.9rem;
        align-items: start;
        padding: 1rem;
        border-radius: 18px;
        background: #eff6ff;
        border: 1px solid var(--line);
        min-width: 0;
    }

    .step-number {
        width: 40px;
        height: 40px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--blue-800), var(--blue-600));
        color: var(--white);
        font-weight: 900;
        box-shadow: 0 10px 20px rgba(37,99,235,0.18);
    }

    .step-item strong {
        display: block;
        margin-bottom: 0.22rem;
        color: var(--blue-950);
        line-height: 1.32;
        overflow-wrap: anywhere;
    }

    .step-item span {
        display: block;
        color: var(--muted);
        line-height: 1.55;
        overflow-wrap: anywhere;
    }

    .model-path {
        padding: 0.95rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #bfdbfe;
        color: var(--blue-950);
        overflow-wrap: anywhere;
        font-weight: 750;
        margin-bottom: 1rem;
    }

    .model-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .model-stat {
        padding: 1rem;
        border-radius: 18px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
    }

    .model-stat strong {
        display: block;
        color: var(--blue-950);
        font-size: 1.2rem;
        margin-bottom: 0.25rem;
    }

    .model-stat span {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .emotion-table-wrap {
        margin-top: 1rem;
        border-radius: 22px;
        overflow: hidden;
        border: 1px solid #bfdbfe;
        background: var(--white);
        box-shadow: 0 14px 34px rgba(37,99,235,0.08);
    }

    .emotion-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }

    .emotion-table thead {
        padding: 0.95rem 1rem;
        background: linear-gradient(135deg, var(--blue-900), var(--blue-700));
        color: var(--white);
        font-weight: 900;
        font-size: 0.92rem;
    }

    .emotion-table th,
    .emotion-table td {
        padding: 0.9rem 1rem;
        text-align: left;
        vertical-align: middle;
    }

    .emotion-table tbody tr {
        border-top: 1px solid var(--line);
        color: var(--ink);
        font-size: 0.94rem;
    }

    .emotion-table tbody tr:nth-child(even) {
        background: #f5f9ff;
    }

    .emotion-table th:first-child,
    .emotion-table td:first-child {
        width: 82px;
    }

    .class-index {
        width: 38px;
        height: 38px;
        border-radius: 13px;
        display: grid;
        place-items: center;
        background: #dbeafe;
        color: var(--blue-800);
        font-weight: 900;
    }

    .class-label-main {
        font-weight: 850;
        color: var(--blue-950);
    }

    .class-label-id {
        display: inline-flex;
        width: fit-content;
        max-width: 100%;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: var(--blue-800);
        font-weight: 850;
        white-space: normal;
    }

    .ui-icon {
        position: relative;
        display: inline-block;
        width: 24px;
        height: 24px;
        color: var(--blue-700);
    }

    .icon-face {
        border: 2px solid currentColor;
        border-radius: 8px;
    }

    .icon-face::before,
    .icon-face::after {
        content: "";
        position: absolute;
        width: 4px;
        height: 4px;
        top: 7px;
        border-radius: 50%;
        background: currentColor;
    }

    .icon-face::before { left: 6px; }
    .icon-face::after { right: 6px; }
    .icon-face i {
        position: absolute;
        left: 7px;
        right: 7px;
        bottom: 6px;
        height: 2px;
        border-radius: 999px;
        background: currentColor;
    }

    .icon-camera {
        width: 25px;
        height: 18px;
        margin-top: 3px;
        border: 2px solid currentColor;
        border-radius: 6px;
    }

    .icon-camera::before {
        content: "";
        position: absolute;
        left: 8px;
        top: 5px;
        width: 6px;
        height: 6px;
        border: 2px solid currentColor;
        border-radius: 50%;
    }

    .icon-camera::after {
        content: "";
        position: absolute;
        left: 4px;
        top: -5px;
        width: 8px;
        height: 5px;
        border-radius: 4px 4px 0 0;
        background: currentColor;
    }

    .icon-shield {
        width: 22px;
        height: 24px;
        border: 2px solid currentColor;
        border-radius: 12px 12px 16px 16px;
    }

    .icon-shield::before {
        content: "";
        position: absolute;
        left: 7px;
        top: 5px;
        width: 6px;
        height: 11px;
        border-right: 2px solid currentColor;
        border-bottom: 2px solid currentColor;
        transform: rotate(38deg);
    }

    .icon-chart {
        border-left: 2px solid currentColor;
        border-bottom: 2px solid currentColor;
    }

    .icon-chart::before,
    .icon-chart::after,
    .icon-chart i {
        content: "";
        position: absolute;
        bottom: 3px;
        width: 4px;
        border-radius: 99px 99px 0 0;
        background: currentColor;
    }

    .icon-chart::before { left: 5px; height: 9px; }
    .icon-chart::after { left: 12px; height: 15px; }
    .icon-chart i { left: 19px; height: 6px; }

    .icon-book {
        width: 24px;
        height: 20px;
        border: 2px solid currentColor;
        border-radius: 5px;
    }

    .icon-book::before {
        content: "";
        position: absolute;
        left: 10px;
        top: 0;
        height: 18px;
        border-left: 2px solid currentColor;
    }

    @media (max-width: 980px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero,
        .camera-grid {
            grid-template-columns: 1fr;
        }

        .hero {
            min-height: auto;
        }

        .visual-panel {
            min-height: 320px;
        }

        .feature-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .info-grid,
        .model-stat-grid {
            grid-template-columns: 1fr;
        }

        .emotion-table th,
        .emotion-table td {
            padding: 0.8rem 0.7rem;
            font-size: 0.88rem;
        }
    }

    @media (max-width: 620px) {
        .hero-actions {
            grid-template-columns: 1fr;
        }

        .feature-grid {
            grid-template-columns: 1fr;
        }

        .floating-result {
            right: 5%;
            top: auto;
            bottom: 6%;
            width: 145px;
        }

        .visual-panel::before {
            inset: 30px;
        }

        .face-preview {
            grid-template-columns: 1fr;
        }

        .prob-row {
            grid-template-columns: 88px minmax(0, 1fr) 52px;
            gap: 0.55rem;
        }
    }
</style>
"""


PAGES = {
    "Beranda": "home",
    "Mulai Deteksi": "detect",
    "Tentang Aplikasi": "about",
    "Cara Penggunaan": "usage",
    "Informasi Model": "model",
}

LOW_CONFIDENCE_THRESHOLD = 0.45


def set_page(page_key):
    st.session_state.current_page = page_key


def icon_html(icon_name):
    return f'<span class="ui-icon {icon_name}"><i></i></span>'


def render_sidebar(model_file):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="side-hero">
                <p>Deteksi emosi wajah berbasis CNN dengan kamera perangkat.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-label">Menu</div>', unsafe_allow_html=True)
        for label, key in PAGES.items():
            button_type = "primary" if st.session_state.current_page == key else "secondary"
            st.button(label, key=f"nav_{key}", type=button_type, on_click=set_page, args=(key,))

        st.markdown('<div class="side-label">Ringkasan</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="side-card">
                Gunakan kamera, ambil foto wajah, lalu baca hasil estimasi emosi dan saran singkat.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if model_file:
            st.markdown(
                f"""
                <div class="side-model">
                    <strong>Model aktif</strong><br>{model_file.name}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="side-model">
                    <strong>Model belum tersedia</strong><br>
                    Letakkan file .keras atau .h5 di folder models/.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_instructions(notebook_files=None):
    notebook_text = ""
    if notebook_files:
        names = ", ".join(file.name for file in notebook_files)
        notebook_text = (
            f"<p>Ditemukan notebook <code>{names}</code> di folder models/. "
            "File .ipynb berisi kode, bukan file model yang bisa langsung dipakai untuk prediksi.</p>"
        )

    st.markdown(
        f"""
        <div class="content-card">
            <h2>File model belum ditemukan</h2>
            <p>Letakkan file model CNN berformat <code>.keras</code> atau <code>.h5</code> di folder <code>models/</code>, lalu jalankan ulang aplikasi.</p>
            {notebook_text}
            <div class="model-path">
                models/<br>
                └── facial_emotion_cnn_best.keras
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_selected_model(model_files):
    if not model_files:
        return None, None, None

    selected_path = model_files[0]

    try:
        model, message = load_emotion_model(selected_path)
        return model, selected_path, message
    except Exception as error:
        return None, selected_path, str(error)


def image_to_data_uri(image):
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_image_card(title, image, caption=None):
    caption_html = f'<p class="face-preview-text">{caption}</p>' if caption else ""
    st.markdown(
        f"""
        <div class="media-card">
            <h3>{title}</h3>
            <img src="{image_to_data_uri(image)}" alt="{title}">
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(probabilities):
    top_index = int(probabilities.argmax())
    rows = []
    for index, value in enumerate(probabilities):
        label = EMOTION_LABELS_ID[EMOTION_LABELS[index]]
        percent = float(value) * 100
        top_class = " is-top" if index == top_index else ""
        rows.append(
            (
                f"<div class='prob-row{top_class}'>"
                f"<div class='prob-label'>{label}</div>"
                "<div class='prob-track'>"
                f"<div class='prob-fill' style='width:{percent:.2f}%'></div>"
                "</div>"
                f"<div class='prob-value'>{percent:.1f}%</div>"
                "</div>"
            )
        )

    return (
        "<div class='prob-card'>"
        "<h3>Probabilitas semua kelas</h3>"
        "<div class='prob-list'>"
        f"{''.join(rows)}"
        "</div>"
        "</div>"
    )


def process_image(image_file, model):
    rgb_image = image_file_to_rgb(image_file)
    face_rgb, face_box = detect_largest_face(rgb_image)
    boxed_image = draw_face_box(rgb_image, face_box)

    if face_rgb is None:
        left, right = st.columns([1, 0.95], gap="large")
        with left:
            render_image_card("Foto dari kamera", rgb_image)
        with right:
            st.markdown(
                """
                <div class="empty-state">
                    <h2>Wajah belum terdeteksi</h2>
                    <p>Wajah belum terdeteksi. Pastikan wajah menghadap kamera dan pencahayaan cukup.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    preprocessed_face, resized_face = prepare_face_for_model(face_rgb)
    emotion_key, confidence, probabilities = predict_emotion(model, preprocessed_face)
    emotion_label = EMOTION_LABELS_ID[emotion_key]
    is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    display_label = "Belum yakin" if is_low_confidence else emotion_label
    suggestion = (
        "Hasil model belum cukup yakin. Coba ambil foto ulang dengan wajah menghadap kamera dan cahaya lebih merata."
        if is_low_confidence
        else EMOTION_SUGGESTIONS[emotion_key]
    )

    media_col, result_col = st.columns([0.92, 1.08], gap="large")
    with media_col:
        render_image_card("Foto dengan area wajah terdeteksi", boxed_image)
        st.markdown(
            f"""
            <div class="media-card">
                <h3>Area wajah yang diproses</h3>
                <div class="face-preview">
                    <img class="face-preview-img" src="{image_to_data_uri(face_rgb)}" alt="Area wajah">
                    <div class="face-preview-text">
                        Area ini diubah menjadi grayscale, di-resize ke 48x48 pixel, lalu dikirim ke model CNN.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-icon">{icon_html("icon-face")}</div>
                <h2>Hasil Deteksi</h2>
                <div class="emotion-name">{display_label}</div>
                <div class="confidence">{confidence * 100:.1f}% tingkat keyakinan</div>
                <div class="suggestion">{suggestion}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if is_low_confidence:
            st.markdown(
                """
                <div class="warning-note">
                    Catatan: confidence rendah sering membuat hasil terlihat sama. Gunakan model yang sudah tervalidasi dan data uji seimbang agar prediksi lebih stabil.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(render_probability_bars(probabilities), unsafe_allow_html=True)


def render_home_page():
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="eyebrow">
                    <span class="ui-icon icon-face"><i></i></span>
                    Kamera wajah untuk mengenali ekspresi siswa
                </div>
                <h1>Deteksi Emosi <span>Wajah Siswa</span> Berbasis CNN</h1>
                <p class="hero-copy">
                    Aplikasi ini membantu mengenali ekspresi wajah siswa dari kamera perangkat.
                    Hasil ditampilkan secara sederhana, rapi, dan mudah dibaca.
                </p>
            </div>
            <div class="visual-panel">
                <div class="scan-corner a"></div>
                <div class="scan-corner b"></div>
                <div class="scan-corner c"></div>
                <div class="scan-corner d"></div>
                <div class="student-avatar"></div>
                <div class="floating-result">
                    <span class="ui-icon icon-face"><i></i></span>
                    <strong>Contoh hasil</strong>
                    <span class="result-percent">92%</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="home-action-gap"></div>', unsafe_allow_html=True)

    col_button, col_note = st.columns([0.24, 0.76])
    with col_button:
        if st.button("Mulai Deteksi", type="primary", use_container_width=True):
            set_page("detect")
            st.rerun()
    with col_note:
        st.markdown(
            '<div class="helper-note">Foto diproses di aplikasi dan digunakan hanya untuk estimasi ekspresi wajah.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon"><span class="ui-icon icon-camera"></span></div>
                <h3>Kamera langsung</h3>
                <p>Siswa cukup menghadap kamera dan mengambil foto wajah.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon"><span class="ui-icon icon-chart"><i></i></span></div>
                <h3>7 kelas emosi</h3>
                <p>Model membaca anger, disgust, fear, happiness, sad, surprised, dan neutral.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon"><span class="ui-icon icon-shield"></span></div>
                <h3>Tampilan aman</h3>
                <p>Hasil disajikan sebagai estimasi, bukan penilaian psikologis.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon"><span class="ui-icon icon-book"></span></div>
                <h3>Saran singkat</h3>
                <p>Setiap hasil dilengkapi saran sederhana untuk mendukung suasana belajar.</p>
            </div>
        </div>
        <div class="disclaimer">
            Hasil deteksi ini hanya estimasi berdasarkan ekspresi wajah dan bukan diagnosis psikologis.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_page(model, model_file, notebook_files):
    st.markdown(
        """
        <div class="page-header">
            <h1 class="page-title">Mulai Deteksi</h1>
            <p class="page-subtitle">
                Izinkan akses kamera, posisikan wajah di tengah, lalu ambil foto. Sistem akan mendeteksi wajah, memotong area wajah, dan menjalankan prediksi model CNN.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        render_model_instructions(notebook_files)
        return

    st.markdown(
        f"""
        <div class="model-path">
            Model aktif: {model_file.name} &nbsp; | &nbsp; Input: grayscale 48x48 pixel
        </div>
        """,
        unsafe_allow_html=True,
    )

    camera_col, info_col = st.columns([1, 1], gap="large")
    with camera_col:
        camera_image = st.camera_input("Ambil foto wajah")
    with info_col:
        st.markdown(
            """
            <div class="content-card tips-card">
                <h2>Tips foto</h2>
                <div class="step-list">
                    <div class="step-item"><div class="step-number">1</div><div><strong>Hadapkan wajah</strong><span>Pastikan wajah mengarah ke kamera dan tidak terlalu jauh.</span></div></div>
                    <div class="step-item"><div class="step-number">2</div><div><strong>Gunakan cahaya cukup</strong><span>Cahaya dari depan membantu deteksi wajah lebih stabil.</span></div></div>
                    <div class="step-item"><div class="step-number">3</div><div><strong>Ambil ulang bila perlu</strong><span>Jika confidence rendah, coba ulang dengan posisi lebih jelas.</span></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if camera_image is not None:
        st.markdown(
            """
            <div class="page-header" style="margin-top:1.2rem;">
                <h1 class="page-title">Hasil Deteksi</h1>
                <p class="page-subtitle">Berikut hasil pemrosesan wajah, estimasi emosi utama, dan probabilitas setiap kelas.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        process_image(camera_image, model)
    else:
        st.markdown(
            """
            <div class="notice-card">
                Ambil foto wajah menggunakan kamera untuk menampilkan hasil deteksi emosi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="disclaimer">
            Hasil deteksi ini hanya estimasi berdasarkan ekspresi wajah dan bukan diagnosis psikologis.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_page():
    st.markdown(
        """
        <div class="page-header">
            <h1 class="page-title">Tentang Aplikasi</h1>
            <p class="page-subtitle">
                Aplikasi ini membantu siswa melihat estimasi ekspresi wajah melalui kamera perangkat dengan tampilan yang sederhana dan mudah dibaca.
            </p>
        </div>
        <div class="content-card">
            <p>
                Sistem menggunakan model CNN yang dilatih dari dataset FER2013. Foto dari kamera diproses untuk mendeteksi area wajah,
                mengubahnya menjadi grayscale 48x48 pixel, lalu menampilkan estimasi emosi dalam bahasa Indonesia.
            </p>
        </div>
        <div class="info-grid">
            <div class="mini-card">
                <div class="feature-icon"><span class="ui-icon icon-camera"></span></div>
                <h3>Berbasis kamera</h3>
                <p>Siswa cukup mengambil foto wajah dari perangkat yang digunakan.</p>
            </div>
            <div class="mini-card">
                <div class="feature-icon"><span class="ui-icon icon-shield"></span></div>
                <h3>Bukan diagnosis</h3>
                <p>Hasil hanya estimasi ekspresi wajah dan perlu dipahami sebagai informasi pendukung.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_usage_page():
    st.markdown(
        """
        <div class="page-header">
            <h1 class="page-title">Cara Penggunaan</h1>
            <p class="page-subtitle">Ikuti langkah sederhana berikut agar kamera dan model dapat membaca wajah dengan lebih baik.</p>
        </div>
        <div class="content-card">
            <div class="step-list">
                <div class="step-item"><div class="step-number">1</div><div><strong>Buka halaman Mulai Deteksi</strong><span>Pilih menu Mulai Deteksi dari sidebar.</span></div></div>
                <div class="step-item"><div class="step-number">2</div><div><strong>Izinkan kamera</strong><span>Berikan izin akses kamera saat browser meminta konfirmasi.</span></div></div>
                <div class="step-item"><div class="step-number">3</div><div><strong>Ambil foto wajah</strong><span>Duduk nyaman, hadapkan wajah ke kamera, dan gunakan pencahayaan yang cukup.</span></div></div>
                <div class="step-item"><div class="step-number">4</div><div><strong>Baca hasil</strong><span>Lihat emosi utama, confidence, grafik probabilitas, dan saran singkat.</span></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_info_page(model_file, load_message):
    model_status = (
        f"<div class='model-path'>Model aktif: {model_file.name}</div>"
        if model_file
        else "<div class='model-path'>Belum ada file model aktif.</div>"
    )

    st.markdown(
        f"""
        <div class="page-header">
            <h1 class="page-title">Informasi Model</h1>
            <p class="page-subtitle">
                Ringkasan model CNN yang digunakan aplikasi untuk membaca ekspresi wajah siswa.
            </p>
        </div>
        <div class="content-card">
            <p class="page-subtitle">
                Model menggunakan input gambar wajah grayscale ukuran 48x48 pixel dengan shape <code>(48, 48, 1)</code>.
                Output model berupa probabilitas untuk 7 kelas emosi FER2013.
            </p>
            {model_status}
        </div>
        <div class="model-stat-grid">
            <div class="model-stat"><strong>48x48</strong><span>Ukuran input wajah setelah preprocessing.</span></div>
            <div class="model-stat"><strong>7 kelas</strong><span>Jumlah label emosi FER2013 yang diprediksi.</span></div>
            <div class="model-stat"><strong>CNN</strong><span>Arsitektur model untuk klasifikasi ekspresi wajah.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if load_message:
        st.info(load_message)

    rows = "".join(
        (
            "<tr>"
            f"<td><span class='class-index'>{index}</span></td>"
            f"<td><span class='class-label-main'>{label}</span></td>"
            f"<td><span class='class-label-id'>{EMOTION_LABELS_ID[label]}</span></td>"
            "</tr>"
        )
        for index, label in EMOTION_LABELS.items()
    )
    table_html = (
        "<div class='content-card'>"
        "<h2>Daftar Kelas Emosi</h2>"
        "<p>Model menghasilkan probabilitas untuk setiap kelas berikut, lalu aplikasi menampilkan kelas dengan nilai tertinggi.</p>"
        "<div class='emotion-table-wrap'>"
        "<table class='emotion-table'>"
        "<thead><tr><th>Index</th><th>Label Model</th><th>Label Tampilan</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    model_files = list_model_candidates()
    notebook_files = list_notebook_candidates()
    model, model_file, load_message = load_selected_model(model_files)

    render_sidebar(model_file)

    page = st.session_state.current_page
    if page == "detect":
        render_detection_page(model, model_file, notebook_files)
    elif page == "about":
        render_about_page()
    elif page == "usage":
        render_usage_page()
    elif page == "model":
        render_model_info_page(model_file, load_message)
    else:
        render_home_page()


if __name__ == "__main__":
    main()
