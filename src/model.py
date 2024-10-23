# -*- coding: utf-8 -*-

import torch.nn.functional as F
from torch import nn
import torch

class SegmentCACSModel():
    
    """
    SegmentCACS model
    """

    def __init__(self, settingsfilepath, overwrite=False):
        """ Init ALMultiTask class
            
        :param settingsfilepath:  Filepath to settings file
        :type settingsfilepath: str
        :param overwrite:  True - overwrite settings file with default settings, False - do not overwrite settings file
        :type overwrite: bool

        """

        self.props = dict(
            NumChannelsIn = 1,
            NumChannelsOut = 2,
            Input_size = (512, 512, 5),
            Output_size = (512, 512, 2),
            device = 'cuda',
            modelname = 'SegmentCACS',
            NumChannelsRegion = 15,
            #NumChannelsMain = 6,
            NumChannelsSegment = 14,
            NumChannelsZero = 2
        )
        #DLBaseModel.__init__(self, settingsfilepath=settingsfilepath, overwrite=overwrite, props=props)
        
    def create(self, params):
        """ Create deep learning model create_CLASS_V01
            
        :param params:  Dictionary of model parameters for model_01
        :type params: dict
        """
            
        props = self.props
        self.params=params

        class Conv_down(nn.Module):
            def __init__(self, in_ch, out_ch, dropout=0.0):
                super(Conv_down, self).__init__()
                self.down = nn.Conv2d(in_ch, out_ch,  kernel_size=4, stride=2, padding=1)
                self.relu1 = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(p=dropout)
                self.conv = nn.Conv2d(out_ch, out_ch,  kernel_size=3, stride=1, padding=1)
                self.norm = nn.BatchNorm2d(out_ch)
                self.relu2 = nn.LeakyReLU(0.2)
                self.down.weight.data.normal_(0.0, 0.1)
                self.conv.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x):
                x = self.down(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x

        class Conv_up(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size_1=3, stride_1=1, padding_1=1, kernel_size_2=3, stride_2=1, padding_2=1, dropout=0.0):
                super(Conv_up, self).__init__()
                self.up = nn.UpsamplingBilinear2d(scale_factor=2)
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size_1, padding=padding_1, stride=stride_1)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size_2, padding=padding_2, stride=stride_2)
                self.relu1 = nn.LeakyReLU(0.2)
                self.relu2 = nn.LeakyReLU(0.2)
                self.dropout = nn.Dropout(p=dropout)
                self.norm = nn.BatchNorm2d(out_ch)
                self.conv1.weight.data.normal_(0.0, 0.1)
                self.conv2.weight.data.normal_(0.0, 0.1)
        
            def forward(self, x1, x2):
                x1 = self.up(x1)
                x = torch.cat((x1, x2), dim=1)
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.norm(x)
                x = self.relu2(x)
                return x
            
        class UNet(nn.Module):
            def __init__(self):
                super(UNet, self).__init__()
                
                num_channels = props['Input_size'][2]+1
                dropout1 = 0.0

                self.conv00 = nn.Conv2d(num_channels, 8, kernel_size=5, padding=2, stride=1)
                self.relu00 = nn.LeakyReLU(0.2)
                self.conv01 = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=1)
                self.relu01 = nn.LeakyReLU(0.2)
                
                self.conv_down1 = Conv_down(16, 16, dropout=dropout1)
                self.conv_down2 = Conv_down(16, 32, dropout=dropout1)
                self.conv_down3 = Conv_down(32, 32, dropout=dropout1)
                self.conv_down4 = Conv_down(32, 64, dropout=dropout1)
                self.conv_down5 = Conv_down(64, 64, dropout=dropout1)
                self.conv_down6 = Conv_down(64, 64, dropout=dropout1)
                self.conv_down7 = Conv_down(64, 128, dropout=dropout1)
                #self.conv_down8 = Conv_down(128, 128, dropout=dropout1)
                
                self.dropout0 = nn.Dropout(p=0.5)
                
                #self.conv_up1_reg = Conv_up(128+128, 128, dropout=dropout1)
                self.conv_up2_reg = Conv_up(128+64, 64, dropout=dropout1)
                self.conv_up3_reg = Conv_up(64+64, 64, dropout=dropout1)
                self.conv_up4_reg = Conv_up(64+64, 64, dropout=dropout1)
                self.conv_up5_reg = Conv_up(64+32, 32, dropout=dropout1)
                self.conv_up6_reg = Conv_up(32+32, 32, dropout=dropout1)
                self.conv_up7_reg = Conv_up(32+16, 32, dropout=dropout1)
                self.conv_up8_reg = Conv_up(32+16, 32, dropout=dropout1)
                self.conv0_reg = nn.Conv2d(32, 24,  kernel_size=3, stride=1, padding=1)
                self.relu0_reg = nn.LeakyReLU(0.2)
                #self.conv1_reg = nn.Conv2d(8, props['NumChannelsRegion'],  kernel_size=3, stride=1, padding=1)
                self.conv1_reg = nn.Conv2d(24, props['NumChannelsRegion'],  kernel_size=3, stride=1, padding=1)
                
                # self.conv_up1_main = Conv_up(128+128+128, 128, dropout=0.01)
                # self.conv_up2_main = Conv_up(128+64+64, 64, dropout=0.01)
                # self.conv_up3_main = Conv_up(64+64+64, 64, dropout=0.01)
                # self.conv_up4_main = Conv_up(64+64+64, 64, dropout=0.01)
                # self.conv_up5_main  = Conv_up(64+32+32, 32, dropout=0.01)
                # self.conv_up6_main  = Conv_up(32+32+32, 32, dropout=0.01)
                # self.conv_up7_main  = Conv_up(32+16+16, 16, dropout=0.01)
                # self.conv_up8_main  = Conv_up(16+16+16+num_channels, 32, dropout=0.01)    
                # self.conv0_main = nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1)
                # self.relu0_main = nn.LeakyReLU(0.2)
                # self.conv1_main = nn.Conv2d(8, props['NumChannelsMain'], kernel_size=3, padding=1, stride=1)

                # self.conv_up1_main = Conv_up(128+128, 128, dropout=0.0)
                # self.conv_up2_main = Conv_up(128+64, 64, dropout=0.0)
                # self.conv_up3_main = Conv_up(64+64, 64, dropout=0.0)
                # self.conv_up4_main = Conv_up(64+64, 64, dropout=0.0)
                # self.conv_up5_main  = Conv_up(64+32, 32, dropout=0.0)
                # self.conv_up6_main  = Conv_up(32+32, 32, dropout=0.0)
                # self.conv_up7_main  = Conv_up(32+16, 16, dropout=0.0)
                # self.conv_up8_main  = Conv_up(16+16+num_channels, 32, dropout=0.0)    
                # self.conv0_main = nn.Conv2d(32, 8, kernel_size=3, padding=1, stride=1)
                # self.relu0_main = nn.LeakyReLU(0.2)
                # self.conv1_main = nn.Conv2d(8, props['NumChannelsMain'], kernel_size=3, padding=1, stride=1)
                
                #self.conv_up1_segment = Conv_up(128+128+128, 64, dropout=0.01)
                #self.conv_up2_segment = Conv_up(64+64+64, 32, dropout=0.01)
                #self.conv_up3_segment = Conv_up(32+64+64, 32, dropout=0.01)
                #self.conv_up3_segment = Conv_up(64+64+64, 32, dropout=dropout1)
                #self.conv_up4_segment = Conv_up(64+64+64, 32, dropout=dropout1)
                #self.conv_up5_segment  = Conv_up(32+32+32, 32, dropout=dropout1)
                self.conv_up6_segment  = Conv_up(32+32+32, 32, dropout=dropout1)
                self.conv_up7_segment  = Conv_up(32+32+16, 32, dropout=dropout1)
                self.conv_up8_segment  = Conv_up(32+32+16+num_channels, 24, dropout=dropout1)  
                self.conv0_segment = nn.Conv2d(24, props['NumChannelsSegment'], kernel_size=3, padding=1, stride=1)
                self.relu0_segment = nn.LeakyReLU(0.2)
                self.conv1_segment = nn.Conv2d(props['NumChannelsSegment'], props['NumChannelsSegment'], kernel_size=3, padding=1, stride=1)

                # self.conv00z = nn.Conv2d(num_channels, 8, kernel_size=5, padding=2, stride=1)
                # self.relu00z = nn.LeakyReLU(0.2)
                # self.conv01z = nn.Conv2d(8, 16, kernel_size=5, padding=2, stride=1)
                # self.relu01z = nn.LeakyReLU(0.2)
                
                self.conv00z = nn.Conv2d(num_channels+14+15, 32, kernel_size=5, padding=2, stride=1)
                self.relu00z = nn.LeakyReLU(0.2)
                self.conv01z = nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1)
                self.relu01z = nn.LeakyReLU(0.2)
                
                self.conv_zero1 = Conv_down(32, 32, dropout=0.0)
                self.conv_zero2 = Conv_down(32, 32, dropout=0.0)
                self.conv_zero3 = Conv_down(32, 32, dropout=0.0)
                self.conv_zero4 = Conv_down(32, 32, dropout=0.0)
                self.conv_zero5 = Conv_down(32, 32, dropout=0.0)
                self.conv_zero6 = Conv_down(32, 32, dropout=0.0)
                self.conv_zero7 = Conv_down(32, 32, dropout=0.0)
                self.conv_zero8 = Conv_down(32, 32, dropout=0.0)
                self.fc_zero = nn.Linear(128, 2)
                #self.soft = nn.Softmax(dim=1)

            def forward(self, x):
                
                x00 = self.conv00(x)
                x00r = self.relu00(x00)
                x01 = self.conv01(x00r)
                x01r = self.relu01(x01)
                
                x1 = self.conv_down1(x01r)
                x2 = self.conv_down2(x1)
                x3 = self.conv_down3(x2)
                x4 = self.conv_down4(x3)
                x5 = self.conv_down5(x4)
                x6 = self.conv_down6(x5)
                x7 = self.conv_down7(x6)
                #x8 = self.conv_down8(x7)
                #x8d = self.dropout0(x8)
                x7d = self.dropout0(x7)
                
                #x9 = self.conv_up1_reg(x8d, x7)
                #x10 = self.conv_up2_reg(x9, x6)
                x10 = self.conv_up2_reg(x7d, x6)
                x11 = self.conv_up3_reg(x10, x5)
                x12 = self.conv_up4_reg(x11, x4)
                x13 = self.conv_up5_reg(x12, x3)
                x14 = self.conv_up6_reg(x13, x2)
                x15 = self.conv_up7_reg(x14, x1)
                x16 = self.conv_up8_reg(x15, x01r)
                xw0 = self.conv0_reg(x16)
                xw1 = self.relu0_reg(xw0)
                xout_region = self.conv1_reg(xw1)

                # x9m = self.conv_up1_main(x8d, torch.cat((x9, x7), dim=1))
                # x10m = self.conv_up2_main(x9m, torch.cat((x10, x6), dim=1))
                # x11m = self.conv_up3_main(x10m, torch.cat((x11, x5), dim=1))
                # x12m = self.conv_up4_main(x11m, torch.cat((x12, x4), dim=1))
                # x13m = self.conv_up5_main(x12m, torch.cat((x13, x3), dim=1))
                # x14m = self.conv_up6_main(x13m, torch.cat((x14, x2), dim=1))
                # x15m = self.conv_up7_main(x14m, torch.cat((x15, x1), dim=1))
                # x16m = self.conv_up8_main(x15m, torch.cat((x16, x01r, x), dim=1))
                # xm0 = self.conv0_main(x16m)
                # xm1 = self.relu0_main(xm0)
                # xout_main = self.conv1_main(xm1)

                #x9s = self.conv_up1_segment(x8d, torch.cat((x9, x7), dim=1))
                #x10s = self.conv_up2_segment(x9s, torch.cat((x10, x6), dim=1))
                #x11s = self.conv_up3_segment(x10s, torch.cat((x11, x5), dim=1))
                #x12s = self.conv_up4_segment(x11, torch.cat((x12, x4), dim=1))
                #x13s = self.conv_up5_segment(x12s, torch.cat((x13, x3), dim=1))
                
                #x13s = self.conv_up5_segment(x4, torch.cat((x13, x3), dim=1))
                
                #x14s = self.conv_up6_segment(x13s, torch.cat((x14, x2), dim=1))
                x14s = self.conv_up6_segment(x13, torch.cat((x14, x2), dim=1))
                x15s = self.conv_up7_segment(x14s, torch.cat((x15, x1), dim=1))
                x16s = self.conv_up8_segment(x15s, torch.cat((x16, x01r, x), dim=1))
                xs0 = self.conv0_segment(x16s)
                xs1 = self.relu0_segment(xs0)
                xout_segment = self.conv1_segment(xs1)
                
                x00z = self.conv00z(torch.cat((x, xout_segment, xout_region), dim=1))
                x00rz = self.relu00z(x00z)
                x01z = self.conv01z(x00rz)
                x01rz = self.relu01z(x01z)
                #print('x01rz', x01rz.shape)
                #print('xout_segment', xout_segment.shape)
                x1s = self.conv_zero1(x01rz)
                x2s = self.conv_zero2(x1s)
                x3s = self.conv_zero3(x2s)
                x4s = self.conv_zero4(x3s)
                x5s = self.conv_zero5(x4s)
                x6s = self.conv_zero6(x5s)
                x7s = self.conv_zero7(x6s)
                x8s = self.conv_zero8(x7s)
                xout_zero = self.fc_zero(torch.flatten(x8s, 1))
                #return xout_region, xout_main, xout_segment, xout_zero
                return xout_region, xout_segment, xout_zero

        unet = UNet()
        unet.train()
        unet.cuda()   
        
        # Create model
        self.model = unet
        #self.count_parameters(self.model)
        #self.opt_unet = optim.Adam(self.model['unet'].parameters(), lr = self.params['lr'], betas=(0.9, 0.999), weight_decay=0.01)


    def load(self, modelpath):
        """
        Load pretained model
        """
        self.model.load_state_dict(torch.load(modelpath))
        
    def predict(self, Xin):
        self.model.eval()
        with torch.no_grad():
            Y_region, Y_lesion = self.model(Xin)
        return Y_region, Y_lesion