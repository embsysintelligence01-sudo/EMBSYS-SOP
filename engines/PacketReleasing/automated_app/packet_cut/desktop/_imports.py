import os
import argparse
import sys
import time
import re
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
import paddle
from paddleocr import PaddleOCR
from ultralytics import YOLO
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame, QMessageBox, QSizePolicy, QProgressBar,
    QScrollArea
)
from packet_cut_core import (
    MIN_BACKGROUND_GAP_PX,
    MIN_NORM_GAP_RATIO,
    MIN_GAP_COVERAGE_RATIO,
    MIN_EXPOSED_GAP_PIXELS,
    MIN_PIECE_AREA_PX,
    MIN_PIECE_AREA_RATIO,
    MIN_TOTAL_AREA_RATIO,
    MAX_TOTAL_AREA_RATIO,
    MAX_CENTROID_SHIFT_PX,
    MAX_AREA_CHANGE_RATIO,
    CUT_VERIFY_SECONDS,
    VERIFY_SAMPLE_INTERVAL,
    MIN_VERIFY_SAMPLES,
    MIN_VALID_SAMPLE_RATIO,
    CLEAR_WAIT_SECONDS,
    HAND_CONFIRM_FRAMES,
    SCISSOR_CONFIRM_FRAMES,
    unpad_letterbox_mask,
    evaluate_hand_detection,
    evaluate_scissor_detection,
    extract_packet_pieces,
    analyze_packet_gap,
    piece_signature,
    is_stable_pair,
    draw_verification_overlay,
    save_verification_debug,
    evaluate_cut_model_prediction,
    evaluate_final_cut_decision
)
