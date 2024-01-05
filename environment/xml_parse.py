import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join

# root = ET.parse('../detector/e2_0.xml').getroot()
onlyfiles = [f for f in listdir('../detector') if isfile(join('../detector', f))]

tags = ['meanSpeed','maxJamLengthInMeters','meanVehicleNumber']
mean = {}
variance = {}

# print(root)
for file in onlyfiles:
    file = '../detector/'+file
    root = ET.parse(file).getroot()
    
    for type_tag in root.findall('interval'):
        for tag in tags:
            value = float(type_tag.get(tag))
            if(tag not in mean):
                mean[tag] = value
                mean['n'+tag] = 1
            else:
                mean[tag] = (mean[tag] * mean['n'+tag] + value) / (mean['n'+tag] + 1)
                mean['n'+tag] = mean['n'+tag] + 1


for file in onlyfiles:
    file = '../detector/'+file
    root = ET.parse(file).getroot()
    for type_tag in root.findall('interval'):
        for tag in tags:
            value = float(type_tag.get(tag))
            if(tag not in variance):
                variance[tag] = (value - mean[tag])**2
                variance['n'+tag] = 1
            else:
                variance[tag] = (variance[tag] * variance['n'+tag] + (value - mean[tag])**2) / (variance['n'+tag] + 1)
                variance['n'+tag] = variance['n'+tag] + 1

print(mean)
print(variance)


            
