import cv2
from utils.callback import callback_per_frame, frame_width, frame_height

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        _ , frame = cap.read()
        frame = callback_per_frame(frame)
        cv2.imshow('cars', frame)

        if cv2.waitKey(30) == 27:
            break


if __name__ == '__main__':
    main()
