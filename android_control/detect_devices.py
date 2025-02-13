import adbutils

class AdbDeviceInfo:
    def __init__(self, host="127.0.0.1", port=5037):
        self.client = adbutils.AdbClient(host=host, port=port)

    def list_devices(self):
        # Correct method to get the list of devices
        devices = self.client.device_list()
        if not devices:
            print("No devices found.")
            return
        
        print("List of connected devices:")
        for device in devices:
            print(f"Serial: {device.serial}, State: {device.get_state()}")

if __name__ == "__main__":
    adb_info = AdbDeviceInfo()
    adb_info.list_devices()
