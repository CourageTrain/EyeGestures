"""
Freeform Calibration + Drawing Mode
- SPACEBAR: Start/Stop Calibration
- 'D' key: Enter Drawing Mode (draw with your eyes)
- 'C' key: Clear the drawing
- CTRL+Q: Quit
"""

import os
import sys
import cv2
import pygame
import numpy as np
import mouse

pygame.init()
pygame.font.init()

screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("EyeGestures - Freeform Calibration + Drawing")

font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)
info_font = pygame.font.Font(None, 24)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3
from eyeGestures.calibration_freeform import FreeformCalibrator

gestures = EyeGestures_v3()
freeform_calibrator = FreeformCalibrator(calibration_radius=1000)
cap = VideoCapture(0)

# Colors
RED = (255, 0, 100)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
LIGHT_BLUE = (100, 150, 255)
PURPLE = (200, 0, 200)

clock = pygame.time.Clock()

# State variables
calibration_active = False
tracking_mode = False
drawing_mode = False
gaze_position = [screen_width // 2, screen_height // 2]
points_collected = 0

# Drawing surface
drawing_surface = pygame.Surface((screen_width - 420, screen_height))
drawing_surface.fill(BLACK)
drawing_points = []  # List of points to draw
last_gaze_position = gaze_position.copy()

running = True
frame_count = 0


def get_raw_eye_features(event_data):
    """Extract raw eye features for calibration"""
    try:
        if event_data is None:
            return None

        if hasattr(event_data, 'point'):
            pt = event_data.point
            if pt is not None and isinstance(pt, (list, tuple, np.ndarray)):
                if len(pt) >= 2:
                    return np.array(pt, dtype=np.float64)

        return None
    except:
        return None


while running:
    frame_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False

            elif event.key == pygame.K_SPACE:
                if drawing_mode:
                    # Exit drawing mode
                    drawing_mode = False
                    print(">>> Exited drawing mode")
                elif not calibration_active:
                    # Start calibration
                    freeform_calibrator.start_calibration()
                    calibration_active = True
                    tracking_mode = False
                    points_collected = 0
                    print("\n>>> CALIBRATION STARTED")
                    print(">>> Follow the yellow cursor with your eyes")
                else:
                    # Stop calibration
                    success = freeform_calibrator.stop_calibration()
                    calibration_active = False
                    print(f"\n>>> CALIBRATION STOPPED - Points: {points_collected}")
                    if success and freeform_calibrator.is_fitted():
                        tracking_mode = True
                        print(">>> CALIBRATION COMPLETE!")
                        print(">>> Press 'D' to enter DRAWING MODE")

            elif event.key == pygame.K_d:
                if tracking_mode and freeform_calibrator.is_fitted():
                    drawing_mode = True
                    drawing_points.clear()
                    drawing_surface.fill(BLACK)
                    print("\n>>> DRAWING MODE ACTIVATED")
                    print(">>> Draw with your eyes!")
                    print(">>> Press SPACEBAR to exit drawing mode")
                    print(">>> Press 'C' to clear drawing")
                else:
                    print(">>> You must complete calibration first! (Press SPACEBAR)")

            elif event.key == pygame.K_c or event.key == pygame.K_C:
                if drawing_mode:
                    drawing_points.clear()
                    drawing_surface.fill(BLACK)
                    print(">>> Drawing cleared!")

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.flip(frame, axis=1)

    event_data = None
    calibration_data = None

    try:
        event_data, calibration_data = gestures.step(
            frame,
            calibration_active,
            screen_width,
            screen_height,
            context="freeform"
        )
    except Exception as e:
        pass

    # ==================== CALIBRATION PHASE ====================
    if calibration_active:
        if event_data is not None:
            mouse_x, mouse_y = mouse.get_position()
            eye_features = get_raw_eye_features(event_data)

            if eye_features is not None:
                screen_point = np.array([mouse_x, mouse_y], dtype=np.float64)
                freeform_calibrator.add_calibration_point(eye_features, screen_point)
                points_collected = freeform_calibrator.points_collected

                if points_collected % 100 == 0:
                    print(f"[Calibration] Points: {points_collected}")

    # ==================== TRACKING PHASE (non-drawing) ====================
    elif tracking_mode and not drawing_mode and freeform_calibrator.is_fitted():
        if event_data is not None:
            eye_features = get_raw_eye_features(event_data)

            if eye_features is not None:
                predicted_gaze = freeform_calibrator.predict(eye_features)

                if predicted_gaze is not None:
                    gaze_x = max(0, min(int(predicted_gaze[0]), screen_width - 1))
                    gaze_y = max(0, min(int(predicted_gaze[1]), screen_height - 1))
                    gaze_position = [gaze_x, gaze_y]

    # ==================== DRAWING PHASE ====================
    elif drawing_mode and tracking_mode and freeform_calibrator.is_fitted():
        if event_data is not None:
            eye_features = get_raw_eye_features(event_data)

            if eye_features is not None:
                predicted_gaze = freeform_calibrator.predict(eye_features)

                if predicted_gaze is not None:
                    gaze_x = max(0, min(int(predicted_gaze[0]), screen_width - 1))
                    gaze_y = max(0, min(int(predicted_gaze[1]), screen_height - 1))
                    gaze_position = [gaze_x, gaze_y]

                    # Add point to drawing if it moved significantly
                    distance = np.sqrt((gaze_position[0] - last_gaze_position[0]) ** 2 +
                                       (gaze_position[1] - last_gaze_position[1]) ** 2)

                    if distance > 3:  # Only draw if moved more than 3 pixels
                        drawing_points.append(tuple(gaze_position))
                        last_gaze_position = gaze_position.copy()

    # ==================== RENDERING ====================
    screen.fill(BLACK)

    # Display camera frame
    if event_data and hasattr(event_data, 'sub_frame') and event_data.sub_frame is not None:
        try:
            frame_surface = pygame.surfarray.make_surface(np.rot90(event_data.sub_frame))
            frame_surface = pygame.transform.scale(frame_surface, (400, 400))
            screen.blit(frame_surface, (0, 0))
        except:
            pass

    # ==================== CALIBRATION MODE UI ====================
    if calibration_active:
        mouse_x, mouse_y = mouse.get_position()

        # Draw cursor circle
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 30, 3)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 15)
        pygame.draw.circle(screen, YELLOW, (mouse_x, mouse_y), 8, 2)

        status_text = "CALIBRATION MODE - Follow the cursor"
        status_surface = bold_font.render(status_text, True, CYAN)
        screen.blit(status_surface, (420, 50))

        points_text = f"Points: {points_collected}"
        points_surface = bold_font.render(points_text, True, GREEN)
        screen.blit(points_surface, (420, 120))

        # Progress bar
        min_points = 10
        progress = min(points_collected / min_points, 1.0)
        bar_width = 300
        bar_height = 20
        pygame.draw.rect(screen, WHITE, (420, 160, bar_width, bar_height), 2)
        pygame.draw.rect(screen, GREEN, (420, 160, int(bar_width * progress), bar_height))

        instr_text = "SPACEBAR: Stop | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 200))

        if freeform_calibrator.is_fitted():
            model_text = "✓ Model FITTED"
            model_surface = info_font.render(model_text, True, GREEN)
        else:
            model_text = "○ Model Training..."
            model_surface = info_font.render(model_text, True, YELLOW)
        screen.blit(model_surface, (420, 240))

    # ==================== TRACKING MODE UI (non-drawing) ====================
    elif tracking_mode and not drawing_mode:
        # Draw gaze tracking dot
        pygame.draw.circle(screen, RED, gaze_position, 20)
        pygame.draw.circle(screen, WHITE, gaze_position, 20, 3)
        pygame.draw.circle(screen, RED, gaze_position, 8)

        status_text = "TRACKING MODE"
        status_surface = bold_font.render(status_text, True, GREEN)
        screen.blit(status_surface, (420, 50))

        gaze_text = f"Position: ({gaze_position[0]}, {gaze_position[1]})"
        gaze_surface = info_font.render(gaze_text, True, WHITE)
        screen.blit(gaze_surface, (420, 120))

        instr_text = "'D': Draw | 'R': Resume Cal | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_text, (420, 160))
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (420, 160))

        cal_quality = freeform_calibrator.get_calibration_quality()
        stats_text = f"Calibration: {cal_quality['points_collected']} points"
        stats_surface = info_font.render(stats_text, True, WHITE)
        screen.blit(stats_surface, (420, 200))

    # ==================== DRAWING MODE UI ====================
    elif drawing_mode:
        # Draw on the drawing surface
        drawing_surface.fill(BLACK)

        # Draw all points
        if len(drawing_points) > 0:
            for i, point in enumerate(drawing_points):
                # Adjust point to drawing surface coordinates (offset by 420)
                surf_x = point[0] - 420
                surf_y = point[1]

                if 0 <= surf_x < drawing_surface.get_width() and 0 <= surf_y < drawing_surface.get_height():
                    pygame.draw.circle(drawing_surface, LIGHT_BLUE, (surf_x, surf_y), 5)

            # Draw lines between consecutive points
            for i in range(1, len(drawing_points)):
                p1 = drawing_points[i - 1]
                p2 = drawing_points[i]

                # Adjust to surface coordinates
                surf_x1 = p1[0] - 420
                surf_y1 = p1[1]
                surf_x2 = p2[0] - 420
                surf_y2 = p2[1]

                if (0 <= surf_x1 < drawing_surface.get_width() and 0 <= surf_y1 < drawing_surface.get_height() and
                        0 <= surf_x2 < drawing_surface.get_width() and 0 <= surf_y2 < drawing_surface.get_height()):
                    pygame.draw.line(drawing_surface, PURPLE, (surf_x1, surf_y1), (surf_x2, surf_y2), 3)

        # Draw current gaze position
        gaze_x = gaze_position[0] - 420
        gaze_y = gaze_position[1]
        if 0 <= gaze_x < drawing_surface.get_width() and 0 <= gaze_y < drawing_surface.get_height():
            pygame.draw.circle(drawing_surface, RED, (gaze_x, gaze_y), 15)
            pygame.draw.circle(drawing_surface, WHITE, (gaze_x, gaze_y), 15, 2)

        # Blit drawing surface to screen
        screen.blit(drawing_surface, (420, 0))

        # Draw UI text
        status_text = "DRAWING MODE - Draw with your eyes!"
        status_surface = bold_font.render(status_text, True, CYAN)
        screen.blit(status_surface, (10, 420))

        points_text = f"Points drawn: {len(drawing_points)}"
        points_surface = info_font.render(points_text, True, GREEN)
        screen.blit(points_surface, (10, 480))

        instr_text = "SPACEBAR: Exit | 'C': Clear | CTRL+Q: Quit"
        instr_surface = info_font.render(instr_text, True, WHITE)
        screen.blit(instr_surface, (10, 520))

    # ==================== IDLE MODE UI ====================
    else:
        idle_text = "READY TO CALIBRATE"
        idle_surface = bold_font.render(idle_text, True, CYAN)
        screen.blit(idle_surface, (420, 50))

        instr1 = "Press SPACEBAR to start calibration"
        instr1_surface = info_font.render(instr1, True, WHITE)
        screen.blit(instr1_surface, (420, 150))

        instr2 = "Follow the yellow cursor with your eyes"
        instr2_surface = info_font.render(instr2, True, WHITE)
        screen.blit(instr2_surface, (420, 190))

        instr3 = "After calibration, press 'D' to draw"
        instr3_surface = info_font.render(instr3, True, WHITE)
        screen.blit(instr3_surface, (420, 230))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
cap.release()
print("\nApplication closed.")