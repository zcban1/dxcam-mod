import ctypes
from dataclasses import dataclass, InitVar
from dxcam._libs.d3d11 import *
from dxcam._libs.dxgi import *
from dxcam.core.device import Device
from dxcam.core.output import Output

@dataclass
class StageSurface:
    width: ctypes.c_uint32
    height: ctypes.c_uint32
    dxgi_format: ctypes.c_uint32 = DXGI_FORMAT_B8G8R8A8_UNORM
    desc: D3D11_TEXTURE2D_DESC = D3D11_TEXTURE2D_DESC()
    texture: ctypes.POINTER(ID3D11Texture2D) = None
    output: InitVar[Output] = None
    device: InitVar[Device] = None

    def __post_init__(self, output, device) -> None:
        self.width, self.height = output.surface_size
        self.desc.Width = self.width
        self.desc.Height = self.height
        self.desc.Format = self.dxgi_format
        self.desc.MipLevels = 1
        self.desc.ArraySize = 1
        self.desc.SampleDesc.Count = 1
        self.desc.SampleDesc.Quality = 0
        self.desc.Usage = D3D11_USAGE_STAGING
        self.desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ
        self.desc.MiscFlags = 0
        self.desc.BindFlags = 0

        self.texture = ctypes.POINTER(ID3D11Texture2D)()
        device.device.CreateTexture2D(
            ctypes.byref(self.desc),
            None,
            ctypes.byref(self.texture),
        )

    def release(self):
        if self.texture:# is not None:
            self.width, self.height = 0, 0
            self.texture.Release()
            self.texture = None

    def map(self):
        #rect: DXGI_MAPPED_RECT = DXGI_MAPPED_RECT()
        rect = DXGI_MAPPED_RECT()
        self.texture.QueryInterface(IDXGISurface).Map(ctypes.byref(rect), 1)
        return rect

    def unmap(self):
        self.texture.QueryInterface(IDXGISurface).Unmap()
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Initialized:{self.texture is not None} Size:{(self.width, self.height)} Format:DXGI_FORMAT_B8G8R8A8_UNORM>"


            self.__class__.__name__,
            self.texture is not None,
            (self.width, self.height),
            "DXGI_FORMAT_B8G8R8A8_UNORM",
        )
