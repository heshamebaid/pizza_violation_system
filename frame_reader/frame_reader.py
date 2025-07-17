import cv2
import pika
import sys
import time
import os

RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'localhost')
RABBITMQ_QUEUE = os.environ.get('RABBITMQ_QUEUE', 'video_frames')


def connect_rabbitmq():
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=RABBITMQ_QUEUE)
    return connection, channel


def publish_frame(channel, frame, frame_id):
    # Encode frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        print(f"Failed to encode frame {frame_id}")
        return
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_QUEUE,
        body=buffer.tobytes(),
        properties=pika.BasicProperties(headers={'frame_id': frame_id})
    )
    print(f"Published frame {frame_id}")


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        sys.exit(1)

    connection, channel = connect_rabbitmq()
    frame_id = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            publish_frame(channel, frame, frame_id)
            frame_id += 1
            time.sleep(1/25)  # Simulate 25 FPS
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        connection.close()
        print("Frame reader stopped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python frame_reader.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
