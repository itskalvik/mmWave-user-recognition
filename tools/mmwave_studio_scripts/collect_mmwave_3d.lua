--Script to collect 100 short(~6 sec) interval data samples with TI mmWaveStudio
--The script also collects video/depth data from a Microsoft Azure Kinect along
--with radar data form TI WR1843 board

user = "subject1"

dataset_path  = "C:\\Users\\mmwave\\Desktop\\mmwave_data\\pose\\"

--ADC_Data file and Raw file and PacketReorder utitlity log file path
lfs              = require "lfs"
buzzer           = "C:\\ti\\mmwave_studio_02_01_00_00\\mmWaveStudio\\Scripts\\buzzer.exe"
camera_recorder  = '"' .. 'C:\\Program Files\\Azure Kinect SDK v1.3.0\\tools\\k4arecorder.exe' .. '"'
raw_file_name    = "adc_data_Raw_"
file_name        = "adc_data"
time_fname       = "timestamps.csv"

RSTD.Sleep(5000)


file_path = dataset_path .. user .. os.date('\\%m_%d\\%H_%M_%S\\')
os.execute("mkdir " .. file_path)

file = io.open(file_path .. time_fname, "a")
io.output(file)

--Start Record ADC data
ar1.CaptureCardConfig_StartRecord(file_path .. file_name .. ".bin", 1)

--Trigger frame
io.popen(camera_recorder .. " -d NFOV_UNBINNED -r 30 --imu OFF -c 1080p -l 597 " .. file_path .. file_name .. ".mkv")
RSTD.Sleep(1500)

io.write(os.time(os.date("!*t")))
ar1.StartFrame()
os.execute(buzzer)
RSTD.Sleep(596000)
io.write(" " .. os.time(os.date("!*t")))
io.close()

for i = 15, 0, -1
do
  size = lfs.attributes(file_path .. raw_file_name .. i .. ".bin" , 'size')
  if((size ~= 1073741760 and i < 15) or (size ~= 851313600 and i == 15)) then
    WriteToLog("Raw_data size:" .. size .. "\n", "red")
    os.execute(buzzer)
    os.execute(buzzer)
    do return end
  end
end

--Signal end of script
os.execute(buzzer)
os.execute(buzzer)
os.execute(buzzer)
