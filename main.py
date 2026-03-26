import os
import cv2
import argparse
import pandas as pd

# ROI по умолчанию, если не выбран ручной режим (--use-select-roi).
DEFAULT_ROI_XYWH = (775, 4, 487, 223)


def parse_args():
    # Аргументы: входное видео, куда писать ролик, опционально эталонный кадр пустого стола(дефолт - 2400 кадр), опционально ручной ROI.
    parser = argparse.ArgumentParser(description="Table cleaning prototype")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/output.mp4", help="Path to output video")
    parser.add_argument("--display", action="store_true", help="Show preview while processing")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of processed frames for debugging")
    parser.add_argument(
        "--empty-frame",
        type=int,
        default=2400,
        help="Frame index where selected table is empty (reference frame)",
    )
    parser.add_argument(
        "--use-select-roi",
        action="store_true",
        help="Select ROI manually with cv2.selectROI instead of hardcoded values",
    )
    return parser.parse_args()


def select_table_roi(first_frame):
    # Пользователь выделяет стол; дальше все сравнения идут только внутри этого прямоугольника.
    roi = cv2.selectROI("Select Table ROI", first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Table ROI")
    return roi


def preprocess_roi(frame, x, y, w, h):
    """Вырезает ROI, переводит в grayscale и слегка размывает."""
    roi = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray


def detect_occupancy(empty_roi_gray, curr_roi_gray, diff_threshold=25, pixel_threshold=1500):
    """
    Сравнение текущего ROI с эталонным пустым ROI.
    Возвращает:
      - occupied_detected: bool
      - changed_pixels: int
      - thresh: бинарная маска отличий
    """
    frame_delta = cv2.absdiff(empty_roi_gray, curr_roi_gray)
    _, thresh = cv2.threshold(frame_delta, diff_threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    changed_pixels = cv2.countNonZero(thresh)
    occupied_detected = changed_pixels > pixel_threshold

    return occupied_detected, changed_pixels, thresh


def detect_activity(prev_roi_gray, curr_roi_gray, diff_threshold=25, pixel_threshold=1500):
    """
    Сравнение ROI текущего и предыдущего кадров.
    Нужна для фиксации движения/подхода к столику.
    """
    frame_delta = cv2.absdiff(prev_roi_gray, curr_roi_gray)
    _, thresh = cv2.threshold(frame_delta, diff_threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    changed_pixels = cv2.countNonZero(thresh)
    activity_detected = changed_pixels > pixel_threshold

    return activity_detected, changed_pixels, thresh


def log_event(events, event_type, frame_idx, fps):
    # Одна строка в журнале: тип события, кадр и время в секундах.
    timestamp_sec = frame_idx / fps
    events.append(
        {
            "event": event_type,
            "frame": frame_idx,
            "timestamp_sec": round(timestamp_sec, 3),
        }
    )


def calculate_empty_to_approach_delays(events_df):
    """
    Для каждого события EMPTY ищет ближайший следующий APPROACH
    и считает задержку.
    """
    delays = []

    if events_df.empty:
        return pd.DataFrame(columns=["empty_time", "approach_time", "delay_sec"])

    empty_events = events_df[events_df["event"] == "empty"].reset_index(drop=True)
    approach_events = events_df[events_df["event"] == "approach"].reset_index(drop=True)

    for _, empty_row in empty_events.iterrows():
        empty_time = empty_row["timestamp_sec"]

        next_approaches = approach_events[approach_events["timestamp_sec"] > empty_time]
        if not next_approaches.empty:
            approach_time = next_approaches.iloc[0]["timestamp_sec"]
            delay_sec = round(approach_time - empty_time, 3)

            delays.append(
                {
                    "empty_time": empty_time,
                    "approach_time": approach_time,
                    "delay_sec": delay_sec,
                }
            )

    return pd.DataFrame(delays)


def ensure_output_dir(path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def prepare_capture_and_reference_roi(args):
    """
    Открывает видео, читает параметры, задаёт ROI (ручной или из константы),
    загружает эталон «пустого стола» и перематывает в начало.
    При ошибке печатает сообщение, освобождает capture и возвращает None.
    """
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    print(f"Размер: {width}x{height}")
    print(f"Кадров: {frame_count}")

    ret, first_frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        cap.release()
        return None

    # Выбор ROI столика: вручную или заранее заданные координаты.
    if args.use_select_roi:
        x, y, w, h = select_table_roi(first_frame)
    else:
        x, y, w, h = DEFAULT_ROI_XYWH

    if w == 0 or h == 0:
        print("ROI не выбран")
        cap.release()
        return None

    print(f"Выбранный ROI: x={x}, y={y}, w={w}, h={h}")

    if args.empty_frame < 0 or args.empty_frame >= frame_count:
        print(f"Некорректный empty-frame: {args.empty_frame}")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.empty_frame)
    ret, empty_frame = cap.read()
    if not ret:
        print("Не удалось прочитать эталонный пустой кадр")
        cap.release()
        return None

    empty_roi_gray = preprocess_roi(empty_frame, x, y, w, h)
    print(f"Эталон пустого стола взят из кадра: {args.empty_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return cap, fps, width, height, x, y, w, h, empty_roi_gray


def analyze_frame_signals(
    prev_roi_gray,
    curr_roi_gray,
    empty_roi_gray,
    reference_weak_threshold,
    reference_strong_threshold,
):
    """
    Сравниваем кусок кадра с эталоном «пусто» и с предыдущим кадром (движение).
    Из отличий от эталона делаем «слабое» и «сильное» срабатывание (пороги настраиваются снаружи).
    """
    _, ref_changed_pixels, _ = detect_occupancy(
        empty_roi_gray,
        curr_roi_gray,
        diff_threshold=25,
        pixel_threshold=reference_weak_threshold,
    )

    reference_weak = ref_changed_pixels > reference_weak_threshold
    reference_strong = ref_changed_pixels > reference_strong_threshold

    activity_detected = False
    motion_changed_pixels = 0

    if prev_roi_gray is not None:
        activity_detected, motion_changed_pixels, _ = detect_activity(
            prev_roi_gray,
            curr_roi_gray,
            diff_threshold=25,
            pixel_threshold=1500,
        )

    next_prev_roi_gray = curr_roi_gray.copy()

    approach_detected = activity_detected
    occupied_detected = reference_strong
    empty_detected = (not activity_detected) and (not reference_weak)

    return (
        ref_changed_pixels,
        reference_weak,
        reference_strong,
        activity_detected,
        motion_changed_pixels,
        next_prev_roi_gray,
        approach_detected,
        occupied_detected,
        empty_detected,
    )


def update_streak_counters(
    occupied_detected,
    approach_detected,
    empty_detected,
    occupied_frames,
    approach_frames,
    empty_frames,
):
    """
    Накапливаем «подряд идущие» кадры по сильнейшему сигналу; если картина смешанная — сбрасываем всё.
    Так убираем дребезг между пусто / движение / занято.
    """
    if occupied_detected:
        occupied_frames += 1
        approach_frames = 0
        empty_frames = 0
    elif approach_detected:
        approach_frames += 1
        occupied_frames = 0
        empty_frames = 0
    elif empty_detected:
        empty_frames += 1
        occupied_frames = 0
        approach_frames = 0
    else:
        occupied_frames = 0
        approach_frames = 0
        empty_frames = 0

    return occupied_frames, approach_frames, empty_frames


def transition_state_machine(
    state,
    frame_idx,
    approach_frames,
    occupied_frames,
    empty_frames,
    approach_threshold_frames,
    occupied_threshold_frames,
    empty_threshold_frames,
    events,
    fps,
):
    """
    Конечный автомат: EMPTY → APPROACH (движение) → OCCUPIED (занято) или пусто по длинной паузе.
    Порядок проверок внутри EMPTY и APPROACH — как в исходнике (сначала подход, потом занято).
    """
    if state == "EMPTY":
        if approach_frames >= approach_threshold_frames:
            state = "APPROACH"
            print(f"[{frame_idx}] EMPTY -> APPROACH")
            log_event(events, "approach", frame_idx, fps)

        elif occupied_frames >= occupied_threshold_frames:
            state = "OCCUPIED"
            print(f"[{frame_idx}] EMPTY -> OCCUPIED")
            log_event(events, "approach", frame_idx, fps)
            log_event(events, "occupied", frame_idx, fps)

    elif state == "APPROACH":
        if occupied_frames >= occupied_threshold_frames:
            state = "OCCUPIED"
            print(f"[{frame_idx}] APPROACH -> OCCUPIED")
            log_event(events, "occupied", frame_idx, fps)

        elif empty_frames >= empty_threshold_frames:
            state = "EMPTY"
            print(f"[{frame_idx}] APPROACH -> EMPTY")

    elif state == "OCCUPIED":
        if empty_frames >= empty_threshold_frames:
            state = "EMPTY"
            print(f"[{frame_idx}] OCCUPIED -> EMPTY")
            log_event(events, "empty", frame_idx, fps)

    return state


def draw_state_overlay(
    frame,
    x,
    y,
    w,
    h,
    state,
    ref_changed_pixels,
    motion_changed_pixels,
    approach_detected,
    occupied_detected,
    reference_weak,
    reference_strong,
):
    # Цвет рамки: зелёный — свободно, жёлтый — движение/подход, красный — занято.
    if state == "EMPTY":
        color = (0, 255, 0)
    elif state == "APPROACH":
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    cv2.putText(
        frame,
        f"state={state}",
        (x, max(y - 35, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"ref_pixels={ref_changed_pixels}",
        (x, max(y - 10, 60)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"motion_pixels={motion_changed_pixels}",
        (x, y + h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"approach={approach_detected} occupied={occupied_detected}",
        (x, y + h + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"ref_weak={reference_weak} ref_strong={reference_strong}",
        (x, y + h + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def finalize_outputs(events_df, args):
    # Печать таблиц, расчёт задержек, сохранение CSV — порядок как раньше.
    if not events_df.empty:
        print("\nСобытия:")
        print(events_df)
    else:
        print("\nСобытия не обнаружены.")

    delays_df = calculate_empty_to_approach_delays(events_df)

    if not delays_df.empty:
        avg_delay = round(delays_df["delay_sec"].mean(), 3)

        print("\nИнтервалы между EMPTY и следующим APPROACH:")
        print(delays_df)

        print(
            f"\nСреднее время между уходом гостя и подходом следующего человека: {avg_delay} сек."
        )
    else:
        avg_delay = None
        print("\nНедостаточно данных для расчета средней задержки.")

    events_csv_path = "outputs/events.csv"
    delays_csv_path = "outputs/delays.csv"

    events_df.to_csv(events_csv_path, index=False)
    delays_df.to_csv(delays_csv_path, index=False)

    print(f"\nСобытия сохранены в: {events_csv_path}")
    print(f"Задержки сохранены в: {delays_csv_path}")
    print(f"Готово. Результат сохранен в: {args.output}")


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Видео не найдено: {args.video}")
        return

    ensure_output_dir(args.output)
    ensure_output_dir("outputs/events.csv")
    ensure_output_dir("outputs/delays.csv")

    prep = prepare_capture_and_reference_roi(args)
    if prep is None:
        return

    cap, fps, width, height, x, y, w, h, empty_roi_gray = prep

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    prev_roi_gray = None
    events = []

    # EMPTY — стол пустой; APPROACH — заметили движение; OCCUPIED — стол занят.
    state = "EMPTY"

    approach_frames = 0
    occupied_frames = 0
    empty_frames = 0

    APPROACH_THRESHOLD_FRAMES = 1

    OCCUPIED_THRESHOLD_FRAMES = int(fps * 1.5)

    # Сколько кадров подряд «тишины» нужно, чтобы снова считать стол пустым.
    EMPTY_THRESHOLD_FRAMES = int(fps * 2.0)

    # Порог по числу отличающихся пикселей от эталона «пусто» (ниже — слабое, выше — сильное).
    REFERENCE_WEAK_THRESHOLD = 3000

    # Сильное отличие от эталона — считаем стол занятым; на ярком солнце может «плавать».
    REFERENCE_STRONG_THRESHOLD = 12000

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if args.max_frames > 0 and frame_idx > args.max_frames:
            print(f"Остановлено по max_frames={args.max_frames}")
            break

        curr_roi_gray = preprocess_roi(frame, x, y, w, h)

        (
            ref_changed_pixels,
            reference_weak,
            reference_strong,
            _,
            motion_changed_pixels,
            prev_roi_gray,
            approach_detected,
            occupied_detected,
            empty_detected,
        ) = analyze_frame_signals(
            prev_roi_gray,
            curr_roi_gray,
            empty_roi_gray,
            REFERENCE_WEAK_THRESHOLD,
            REFERENCE_STRONG_THRESHOLD,
        )

        occupied_frames, approach_frames, empty_frames = update_streak_counters(
            occupied_detected,
            approach_detected,
            empty_detected,
            occupied_frames,
            approach_frames,
            empty_frames,
        )

        state = transition_state_machine(
            state,
            frame_idx,
            approach_frames,
            occupied_frames,
            empty_frames,
            APPROACH_THRESHOLD_FRAMES,
            OCCUPIED_THRESHOLD_FRAMES,
            EMPTY_THRESHOLD_FRAMES,
            events,
            fps,
        )

        draw_state_overlay(
            frame,
            x,
            y,
            w,
            h,
            state,
            ref_changed_pixels,
            motion_changed_pixels,
            approach_detected,
            occupied_detected,
            reference_weak,
            reference_strong,
        )

        writer.write(frame)

        if args.display:
            cv2.imshow("Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Остановлено пользователем")
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    events_df = pd.DataFrame(events)

    finalize_outputs(events_df, args)


if __name__ == "__main__":
    main()
