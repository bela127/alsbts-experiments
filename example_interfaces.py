#--- Interface:
class USB():
    def transfer(self, data: str) -> None:
        pass
#--- Instanziierungen: (bieten es an)

class USBDisplay(USB):
    def __init__(self, text_style) -> None:
        super().__init__()

    def transfer(self, data):
        print(data)
        return super().transfer(data)
    
class USBStick(USB):
    def __init__(self) -> None:
        super().__init__()

    def transfer(self, data):
        print(f"fast writing to stick: {data}")
        return super().transfer(data)

class USBHDD(USB):
    def transfer(self, data, fake):
        print(f"writing to disk: {data}")
        print(f"still writing: {data}")
        return super().transfer(data)
    
#--- Nuzer:

class Computer():

    def __init__(self, usb_anschluss: USB) -> None:
        self.usb_anschluss = usb_anschluss

    def write_data_to(self, usb_anschluss: USB):
        usb_anschluss.transfer("Hallo")

    def write_data_to_usb(self):
        self.usb_anschluss.transfer("Hallo")

#--- etwas soll am beginn mit dem computer gekoppelt werden und wird dann immer wieder verwendet:
computer = Computer(USBDisplay(text_style="uppercase")) # Das hier ist ein computer blueprint!!


computer.write_data_to_usb()
computer.write_data_to_usb()

computer.write_data_to(USBStick())