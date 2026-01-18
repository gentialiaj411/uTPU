import argparse
import serial


def main():
    parser = argparse.ArgumentParser(description="UART RX probe (read a byte)")
    parser.add_argument("--port", "-p", required=True, help="Serial port, e.g. /dev/ttyUSB1")
    parser.add_argument("--baud", "-b", type=int, default=115200)
    parser.add_argument("--timeout", "-t", type=float, default=1.0)
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    data = ser.read(1)
    if data:
        print(f"RX: {data.hex()}")
    else:
        print("RX: (no data)")
    ser.close()


if __name__ == "__main__":
    main()
