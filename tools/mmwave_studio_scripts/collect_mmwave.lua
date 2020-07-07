--Script to collect 100 short(~6 sec) interval data samples with TI mmWaveStudio
user = "subject1"

dataset_path  = "C:\\Users\\mmwave\\Desktop\\mmwave_data\\conf\\"

--ADC_Data file and Raw file and PacketReorder utitlity log file path
adc_data_path = "C:\\ti\\mmwave_studio_02_01_00_00\\mmWaveStudio\\PostProc\\adc_data.bin"
Raw_data_path = "C:\\ti\\mmwave_studio_02_01_00_00\\mmWaveStudio\\PostProc\\adc_data_Raw_0.bin"
buzzer        = "C:\\ti\\mmwave_studio_02_01_00_00\\mmWaveStudio\\Scripts\\buzzer.exe"
lfs           = require "lfs"

os.execute("mkdir " .. dataset_path .. user .. os.date('\\%m_%d\\'))

--get number of files collected so far
os.execute("dir " .. dataset_path .. user .. os.date('\\%m_%d\\') .. "*.bin /b | find /c /v \"::\" > " .. dataset_path .. "tmp")
for line in io.lines(dataset_path .. "tmp") do count = line end
os.execute("del " .. dataset_path .. "tmp")

RSTD.Sleep(5000)

for i = 100-count,1,-1
do
    WriteToLog("Samples Left:" .. i .. "\n", "red")

    --Start Record ADC data
    ar1.CaptureCardConfig_StartRecord(adc_data_path, 1)
    RSTD.Sleep(1000)

    --Trigger frame
    file_path = user .. os.date('\\%m_%d\\') .. os.date('\\%H_%M_%S')
    ar1.StartFrame()
    os.execute(buzzer)
    RSTD.Sleep(9000)

    --Check file size
    size = lfs.attributes(Raw_data_path , 'size')
    if(size ~= 188416000) then
      WriteToLog("Raw_data size:" .. size .. "\n", "red")
      os.execute(buzzer)
      os.execute(buzzer)
      do return end
    end

    os.rename(Raw_data_path, dataset_path .. file_path .. ".bin")
end

--Signal end of script
os.execute(buzzer)
os.execute(buzzer)
os.execute(buzzer)
