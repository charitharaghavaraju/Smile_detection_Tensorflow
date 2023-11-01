import cv2
from tensorflow import keras
import numpy as np
from model import build_model


def main():
    # capture frame from webcam video. press 'q' to select a frame
    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    w, h, _ = frame.shape

    # reshape frame to meet model input size
    image = cv2.resize(frame, (64, 64))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.expand_dims(image_rgb, 0)

    modelx = build_model()
    modelx.load_weights('best_model2.h5')
    pred = modelx.predict(image_rgb)

    if pred >= 0.5:
        cv2.putText(frame, 'Smile', fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 255, 0),
                    org=(int(h / 2) - 100, int(w / 2) - 100), thickness=4, fontScale=4)
    else:
        cv2.putText(frame, 'Not Smiling', fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 255),
                    org=(int(h / 2) - 100, int(w / 2) - 100), thickness=4, fontScale=4)

    cv2.imwrite('webcam_img_predicted1.jpg', frame)
    cv2.imshow('webcam image', frame)
    cv2.waitKey(5000)


if __name__ == '__main__':
    main()
