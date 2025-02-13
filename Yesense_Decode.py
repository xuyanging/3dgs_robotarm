# -*- encoding:utf-8 -*-
#!/usr/bin/env python3

import serial
import time

#class yesense_product:
#	def __init__(self, model):
yis_out = {'tid':1, 'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
			'q0':1.0, 'q1':1.0, 'q2':0.0, 'q3':0.0, \
			'sensor_temp':25.0, 'acc_x':0.0, 'acc_y':0.0, 'acc_z':1.0, \
			'gyro_x':0.0, 'gyro_y':0.0, 'gyro_z':0.0,	\
			'norm_mag_x':0.0, 'norm_mag_y':0.0, 'norm_mag_z':0.0,	\
			'raw_mag_x':0.0, 'raw_mag_y':0.0, 'raw_mag_z':0.0,	\
			'lat':0.0, 'longt':0.0, 'alt':0.0,	\
			'vel_e':0.0, 'vel_n':0.0, 'vel_u':0.0, \
			'ms':0, 'year': 2022, 'month':8, 'day': 31, \
			'hour':12, 'minute':0, 'second':0,	\
			'samp_timestamp':0, 'dataready_timestamp':0 ,'status':0
			}

#little-endian
YIS_HEADER_1ST	= 0x59
YIS_HEADER_2ND 	= 0x53					

#header(2B) + tid(2B) + len(1B) + CK1(1B) + CK2(1B)
PROTOCOL_MIN_LEN 			= 7	

PROTOCOL_TID_LEN 			= 2
PROTOCOL_PAYLOAD_LEN		= 1
PROTOCOL_CHECKSUM_LEN		= 2
PROTOCOL_TID_POS 			= 2
PROTOCOL_PAYLOAD_LEN_POS	= 4
CRC_CALC_START_POS	        = 2
PAYLOAD_POS					= 5

TLV_HEADER_LEN				= 2		#type(1B) + len(1B)

#输出数据的data id定义
sensor_temp_id 	= 0x01				#data_id
acc_id 			= 0x10				
gyro_id 		= 0x20
norm_mag_id		= 0x30
raw_mag_id		= 0x31
euler_id		= 0x40
quaternion_id	= 0x41
utc_id			= 0x50
samp_timestamp_id  = 0x51
dataready_timestamp_id = 0x52
location_id		= 0x68
speed_id		= 0x70
status_id		= 0x80

#输出数据的len定义
sensor_temp_len	= 0x02				#data_id
acc_len			= 0x0C				
gyro_len 		= 0x0C
norm_mag_len	= 0x0C
raw_mag_len		= 0x0C
euler_len		= 0x0C
quaternion_len	= 0x10
utc_len			= 0x0B
samp_timestamp_len = 0x04
dataready_timestamp_len = 0x04
location_len	= 0x14
speed_len		= 0x0C
status_len		= 0x01

norm_mag_xx=0
norm_mag_yy=0
norm_mag_zz=0

data_factor_not_raw_mag 	= 0.000001	#非原始磁力输出数据与真实数据的转换系数
data_factor_raw_mag			= 0.001		#原始磁力输出数据与真实数据的转换系数
data_factor_sensor_temp 	= 0.01
data_factor_high_res_loc 	= 0.0000000001
data_factor_alt				= 0.001
data_factor_speed			= 0.001

buf 	= [0] * 512					#解析缓冲区
buf_len = 0;						#解析缓冲区数据长度

##--------------------------------------------------------------------------------------##
##                                   解析主函数                                         	##
##   输入：data -- 当前获取的传感器数据，num -- 当前获取的传感器数据长度                			##
##   输入：info -- 存放解析完成后的真实数据字典，debug_flg -- 调试标志，True打开        			##
##   返回值：True -- 解析成功，False -- 解析不成功，包括crc校验错误等情况               			##
##--------------------------------------------------------------------------------------##		
def decode_data(data, num, info, debug_flg):
	global buf
	global buf_len
	
	pos = 0	#记录当前解析的数据的位置
	cnt = 0
	data_len = 0
	check_sum = 0
	
	#更新本次读取的数据到解析缓冲区
	buf[buf_len : buf_len + num] = data[0 : num]
	buf_len += num
	if(debug_flg):
		print('cur_num = ', num)
		print('buf_len = ', buf_len)

	#待解析总长度不足最小一帧数据
	if (buf_len < PROTOCOL_MIN_LEN):
		if(debug_flg):
			print('len not enough')		
		return False
		
	cnt = buf_len
	
	#第一步 -- 查找帧头
	while(cnt > 0):
		if (YIS_HEADER_1ST == buf[pos]):
			if (YIS_HEADER_2ND == buf[pos + 1]) :
				break;	#找到帧头
		cnt -= 1
		pos += 1
			
	if(debug_flg):
		print('start pos = ', pos)	
		
	if (cnt < PROTOCOL_MIN_LEN):
		if(debug_flg):
			print('clear_data')	
		clear_data(pos)
		return False
	
	#查找到帧头，且剩余帧长大于最小一帧数据长度，开始解析
	data_len = buf[pos + PROTOCOL_PAYLOAD_LEN_POS]
	if(debug_flg):
		print('payload_len = ', data_len)			
	
	#校验有效总字节长度和协议字节长度对比
	if (PROTOCOL_MIN_LEN + data_len > cnt):
		if(debug_flg):		
			print('protocol len is not enough')	
		clear_data(pos)	
		return False

	if(debug_flg):		
		print('start calc checksum, pos = ', pos)

	#计算校验和
	check_sum = calc_checksum(buf[pos + CRC_CALC_START_POS : buf_len], CRC_CALC_LEN(data_len))
	if (check_sum != (buf[pos + PROTOCOL_CRC_DATA_POS(data_len)] + (buf[pos + PROTOCOL_CRC_DATA_POS(data_len) + 1] << 8))):
		clear_data(pos + data_len + PROTOCOL_MIN_LEN)
		return False
	
	if(debug_flg):
		print('checksum done')		
	info['tid'] = buf[pos + PROTOCOL_TID_POS] + (buf[pos + PROTOCOL_TID_POS + 1] << 8)
	cnt = data_len
	
	#解析数据
	pos += PAYLOAD_POS;	#payload数据起始位置
	tlv = {'id':0x10, 'len':0x0C}		#每个数据包由data_id + data_len + data组成
	while (data_len > 0 and pos <= buf_len):
		tlv['id'] = buf[pos]
		tlv['len'] = buf[pos + 1]
		ret = parse_data_by_id(tlv, buf[pos + TLV_HEADER_LEN: buf_len], info, debug_flg)
		if(debug_flg):
			print(tlv)
			print('ret = ', ret)
			print('parse data_len %d, pos' %data_len, pos)			
		if(True == ret):
			pos += tlv['len'] + TLV_HEADER_LEN
			data_len -= tlv['len'] + TLV_HEADER_LEN
		else:
			pos += 1
			data_len -= 1
			
	if(debug_flg):			
		print('total len : ', buf_len)
		
	clear_data(pos + PROTOCOL_CHECKSUM_LEN)
	if(debug_flg):
		print('anlysis done, pos = %d, buf_len left %d' %(pos, buf_len));
	
	return True

##--------------------------------------------------------------------------------------##
##                                  缓冲区处理函数                                      	##
##   输入：clr_len -- 待清除的数据长度，从缓冲区第一个数据开始算                        			##
##   返回值：无                                                                         	##
##--------------------------------------------------------------------------------------##			
def clear_data(clr_len):
	global buf
	global buf_len
	
	if(0 == clr_len):
		return
		
	buf[0:clr_len] = [0]	
	if (buf_len > clr_len):
		buf[0 : buf_len - clr_len] = buf[clr_len : buf_len]
		buf[clr_len : buf_len] = [0]
		buf_len -= clr_len		
	else:
		buf_len = 0;	

##--------------------------------------------------------------------------------------##
##                                   数据解析函数                                       	##
##   输入：tlv -- 当前待解析数据头信息，payload -- 当前待解析数据的有效载荷             			##
##   输入：存放解析完成后的真实数据字典，debug_flg -- 调试标志，True打开                			##
##   返回值：True -- 解析成功，False -- 未识别的数据                                    		##
##--------------------------------------------------------------------------------------##				
def parse_data_by_id(tlv, payload, info, debug_flg):
	ret = True
	if (sensor_temp_id == tlv['id'] and sensor_temp_len == tlv['len']):
		if(debug_flg):
			print('data temp')
		info['sensor_temp'] = get_int16_lit(payload) * data_factor_sensor_temp
	elif (acc_id == tlv['id'] and acc_len == tlv['len']):
		if(debug_flg):
			print('data acc')
		info['acc_x'] = get_int32_lit(payload) * data_factor_not_raw_mag		
		info['acc_y'] = get_int32_lit(payload[4:8]) * data_factor_not_raw_mag	
		info['acc_z'] = get_int32_lit(payload[8:12]) * data_factor_not_raw_mag			
	elif (gyro_id == tlv['id'] and gyro_len == tlv['len']):
		if(debug_flg):
			print('data gyro')
		info['gyro_x'] = get_int32_lit(payload) * data_factor_not_raw_mag		
		info['gyro_y'] = get_int32_lit(payload[4:8]) * data_factor_not_raw_mag	
		info['gyro_z'] = get_int32_lit(payload[8:12]) * data_factor_not_raw_mag		
	elif (euler_id == tlv['id'] and euler_len == tlv['len']):
		if(debug_flg):
			print('data euler')
		info['pitch'] = get_int32_lit(payload) * data_factor_not_raw_mag		
		info['roll'] = get_int32_lit(payload[4:8]) * data_factor_not_raw_mag	
		info['yaw'] = get_int32_lit(payload[8:12]) * data_factor_not_raw_mag		
	elif (quaternion_id == tlv['id'] and quaternion_len == tlv['len']):
		if(debug_flg):	
			print('data quaternion')
		info['q0'] = get_int32_lit(payload) * data_factor_not_raw_mag		
		info['q1'] = get_int32_lit(payload[4:8]) * data_factor_not_raw_mag	
		info['q2'] = get_int32_lit(payload[8:12]) * data_factor_not_raw_mag		
		info['q3'] = get_int32_lit(payload[12:16]) * data_factor_not_raw_mag				
	elif (norm_mag_id == tlv['id'] and norm_mag_len == tlv['len']):
		global norm_mag_xx, norm_mag_yy, norm_mag_zz
		norm_mag_xx = get_int32_lit(payload) * data_factor_not_raw_mag		
		norm_mag_yy = get_int32_lit(payload[4:8]) * data_factor_not_raw_mag	
		norm_mag_zz = get_int32_lit(payload[8:12]) * data_factor_not_raw_mag		
		
		if(debug_flg):	
			print('data norm mag')
		info['norm_mag_x'] = get_int32_lit(payload) * data_factor_not_raw_mag		
		info['norm_mag_y'] = get_int32_lit(payload[4:8]) * data_factor_not_raw_mag	
		info['norm_mag_z'] = get_int32_lit(payload[8:12]) * data_factor_not_raw_mag		
	elif (raw_mag_id == tlv['id'] and raw_mag_len == tlv['len']):
		if(debug_flg):	
			print('data raw mag')
		info['raw_mag_x'] = get_int32_lit(payload) * data_factor_raw_mag		
		info['raw_mag_y'] = get_int32_lit(payload[4:8]) * data_factor_raw_mag	
		info['raw_mag_z'] = get_int32_lit(payload[8:12]) * data_factor_raw_mag		
	elif (location_id == tlv['id'] and location_len == tlv['len']):
		if(debug_flg):	
			print('data location')
		info['alt'] = get_int64_lit(payload) * data_factor_high_res_loc		
		info['longt'] = get_int64_lit(payload[8:16]) * data_factor_high_res_loc	
		info['alt'] = get_int32_lit(payload[16:20]) * data_factor_alt		
	elif (utc_id == tlv['id'] and utc_len == tlv['len']):
		if(debug_flg):	
			print('data utc')
		info['ms'] = get_int32_lit(payload)		
		info['year'] = get_int32_lit(payload)	
		info['month'] = get_int32_lit(payload)		
		info['day'] = get_int32_lit(payload)	
		info['hour'] = get_int32_lit(payload)		
		info['minute'] = get_int32_lit(payload)	
		info['second'] = get_int32_lit(payload)			
	elif (samp_timestamp_id == tlv['id'] and samp_timestamp_len == tlv['len']):
		if(debug_flg):	
			print('data samp_timestamp')
		info['samp_timestamp'] = get_int32_lit(payload)		
	elif (speed_id == tlv['id'] and speed_len == tlv['len']):
		if(debug_flg):	
			print('data speed')
		info['vel_e'] = get_int32_lit(payload) * data_factor_speed		
		info['vel_n'] = get_int32_lit(payload[4:8]) * data_factor_speed	
		info['vel_u'] = get_int32_lit(payload[8:12]) * data_factor_speed	
	elif (dataready_timestamp_id == tlv['id'] and dataready_timestamp_len == tlv['len']):
		if(debug_flg):	
			print('data  dataready_timestamp')
		info['dataready_timestamp'] = get_int32_lit(payload)
	elif (status_id == tlv['id'] and status_len == tlv['len']):
		if(debug_flg):	
			print('data fusion status')
		info['status'] = payload[0]			
	else:
		print('unknown data id && len')
		ret = False
		
	return ret

##--------------------------------------------------------------------------------------##
##                       计算待计算CRC的数据长度的函数                                  		##
##   输入：payload_len -- 当前数据帧的有效载荷长度                                      		##
##   返回值：待计算CRC的总数据长度                                                      		##
##--------------------------------------------------------------------------------------##		
def CRC_CALC_LEN(payload_len):	#3 = tid(2B) + len(1B) 	
	return (payload_len + PROTOCOL_TID_LEN + PROTOCOL_PAYLOAD_LEN)	

##--------------------------------------------------------------------------------------##
##                       计算输出报文数据中CRC的位置的函数                              		##
##   输入：payload_len -- 当前数据帧的有效载荷长度                                      		##
##   返回值：当前输出的数据报文中CRC的起始位置                                          		##
##--------------------------------------------------------------------------------------##			
def PROTOCOL_CRC_DATA_POS(payload_len):
	return (CRC_CALC_START_POS + CRC_CALC_LEN(payload_len))
	
##--------------------------------------------------------------------------------------##
##                                   CRC计算函数                                        	##
##   输入：data -- 待计算CRC的数据, len - 待计算CRC数据的长度                           		##
##   返回值：CRC计算结果，占2字节长度                                                   		##
##--------------------------------------------------------------------------------------##			
def calc_checksum(data, len):
	check_a = 0x00
	check_b = 0x00

	for i in range(0, len):
		check_a += data[i]
		check_b += check_a
	return ((check_b % 256) << 8) + (check_a % 256)

##--------------------------------------------------------------------------------------##
##                      将8bit数据流转换为有符号的16bit数据函数                         		##
##   输入：data -- 待计算的8bit数据流                                                   	##
##   返回值：转换后的16bit的转换结果                                                    		##
##--------------------------------------------------------------------------------------##			
def get_int16_lit(data):
	temp = 0

	temp = data[0]
	temp += data[1] << 8
	
	if(temp & 0x8000):
		temp -= 1
		temp = ~temp
		temp &= 0x7FFF
		temp = 0 - temp
	
	return temp

##--------------------------------------------------------------------------------------##
##                      将8bit数据流转换为有符号的32bit数据函数                           	##
##   输入：data -- 待计算的8bit数据流                                                   	##
##   返回值：转换后的32bit的转换结果                                                    		##
##--------------------------------------------------------------------------------------##			
def get_int32_lit(data):
	temp = 0
	if len(data) < 4:
		raise ValueError("Not enough data to extract 32-bit integer")

	for i in range(0, 4):
		temp += data[i] << (i * 8)

	if(temp & 0x8000_0000):
		temp -= 1
		temp = ~temp
		temp &= 0x7FFFFFFF
		temp = 0 - temp
			
	return temp

##--------------------------------------------------------------------------------------##
##                      将8bit数据流转换为有符号的64bit数据函数                         		##
##   输入：data -- 待计算的8bit数据流                                                   	##
##   返回值：转换后的64bit的转换结果                                                    		##
##--------------------------------------------------------------------------------------##			
def get_int64_lit(data):
	temp = 0
	if len(data) < 8:
		raise ValueError("Not enough data to extract 64-bit integer")
	
	for i in range(0, 8):
		temp += data[i] << (i * 8)

	if(temp & 0x8000_0000_0000_0000):
		temp -= 1
		temp = ~temp
		temp &= 0x7FFF_FFFF_FFFF_FFFF
		temp = 0 - temp
			
	return temp
	
##--------------------------------------------------------------------------------------##
##                                   打开串口函数                                         ##
##   输入：port -- 待打开的串口号，字符串，baud -- 待打开串口的波特率，整数                       ##
##   返回值：串口示例                                                                      ##
##--------------------------------------------------------------------------------------##			
def open_serial(port, baud):
	ser = serial.Serial(port, baud, timeout = 0.002)
	if (ser.is_open):
		print("open %s with baudrate %d suc!" %(port, baud))
	else:
		print("open %s with baudrate %d fail!" %(port, baud))
	return ser
	
##--------------------------------------------------------------------------------------
if __name__ == '__main__':
	try:
		#conn = open_serial('com11', 460800)		#windows下为comxx，linux下为/dev/ttySCx或/dev/ttyUSBx
		conn=open_serial('/dev/ttyACM0', 460800)
	except Exception as e:
		print('-------Exception------: ', e)

	while(1):
		data = conn.read_all()
		#data = conn.read(conn.in_waiting)
		num = len(data)
		if(num > 0):
			ret = decode_data(data, num, yis_out, False)
			if(True == ret):
				print('tid %d, pitch %f, roll %f, yaw %f, acc_x %f, acc_y %f, acc_z %f, gyro_x %f, gyro_y %f, gyro_z %f,timestamp %d' \
				%(yis_out['tid'], yis_out['pitch'], yis_out['roll'], yis_out['yaw'], \
				yis_out['acc_x'], yis_out['acc_y'], yis_out['acc_z'],\
				yis_out['gyro_x'], yis_out['gyro_y'], yis_out['gyro_z'],yis_out['samp_timestamp']))
				print("norm_mag_x="+str(norm_mag_xx))
				print("norm_mag_y="+str(norm_mag_yy))
				print("norm_mag_z="+str(norm_mag_zz))
		# time.sleep(0.005)   #windows下需去掉，否则运行不正常
