filename$ = "dat_append"

Read Table from comma-separated file... 'filename$'.csv

selectObject: "Table 'filename$'"
tableCols = Get number of rows

writeInfoLine: "Start"
for i from 1 to tableCols
	#appendInfoLine: i

#Get time information...........................................
selectObject: "Table 'filename$'"

	tmpFile$ = Get value... i speaker
	tmpTime = Get value... i time

#Get pitch......................................................
selectObject: "Pitch " + tmpFile$
	
	tmpPitchValue = Get value at time... tmpTime Hertz Linear

#Get intensity..................................................
selectObject: "Intensity " + tmpFile$
	
	tmpIntensityValue = Get value at time... tmpTime Cubic

#Get formants...................................................
selectObject: "Formant " + tmpFile$
	
	f1Value = Get value at time... 1 tmpTime hertz Linear
	f2Value = Get value at time... 2 tmpTime hertz Linear
	f3Value = Get value at time... 3 tmpTime hertz Linear

#Get mfccs...................................................
selectObject: "MFCC " + tmpFile$

	frame_number = Get frame number from time... tmpTime

#Insert values in the table.....................................
selectObject: "Table 'filename$'"

	Set numeric value... i pitchValue tmpPitchValue

	Set numeric value... i intensityValue tmpIntensityValue

	Set numeric value... i formant_1 f1Value
	Set numeric value... i formant_2 f2Value
	Set numeric value... i formant_3 f3Value

endfor

selectObject: "Table 'filename$'"

Save as comma-separated file... 'filename$'.csv

appendInfoLine: "End"






