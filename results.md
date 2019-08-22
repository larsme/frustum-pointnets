old false negatives: [4210724 4210724 4210724]
new false negatives: [104821 104821 104821]

TODO:
recreate train, val datasets

with ideal 2D boxes
	200 Epochen gegebenes V1
		Segmentation results:
		['Car', 'Pedestrian', 'Cyclist']
		IOU:
		[ 0.8102318   0.795288    0.80539635]
		Precision:
		[ 0.860192    0.87287648  0.90955295]
		Recall
		[ 0.93311127  0.89946771  0.87551608]
		instance IOU:
		[ 0.80141063  0.79942365  0.80782508]
		instance Precision:
		[ 0.84857318  0.86508555  0.89705101]
		instance Recall
		[ 0.92344048  0.87913487  0.8712751 ]




	6 Epochen, V1
		Segmentation results:
		['Car', 'Pedestrian', 'Cyclist']
		IOU:
		[ 0.73068229  0.60175574  0.64546482]
		Precision:
		[ 0.82811927  0.7869503   0.87456565]
		Recall
		[ 0.86130541  0.71886821  0.71131527]
		instance IOU:
		[ 0.72287842  0.60478548  0.65102351]
		instance Precision:
		[ 0.81116224  0.69888909  0.78161011]
		instance Recall
		[ 0.85500223  0.71845766  0.71658276]

	206 Epochen, V1
		Segmentation results:
		['Car', 'Pedestrian', 'Cyclist']
		IOU:
		[0.81037337 0.79457019 0.8068369 ]
		Precision:
		[0.86026001 0.87266486 0.90933549]
		Recall
		[0.933219   0.89877399 0.87742101]
		instance IOU:
		[0.80153775 0.7990387  0.8092921 ]
		instance Precision:
		[0.84869415 0.86457761 0.89637012]
		instance Recall
		[0.92359421 0.87871946 0.87427532]





Statistics points per box:
				Train						Val
	Car
		boxes
			14342							16314
		mean
			624.5170826941849				701.8325977687875		
		var
			618331.9006634179				727071.7658345292
	Pedestrian
		boxes
			4372							5254
		mean
			300.1633119853614				359.2116482679863
		var
			154888.28256981762				216417.40705122307
	Cyclist
		boxes
			2958							3824
		mean
			448.9871534820825				482.3258368200837
		var
			355310.9159945343				392908.00313999015
	Ges
		boxes
			21672							25392
		mean
			535.1255998523441				597.8815768746061
		var
			506377.6321574937				591824.9778237893
	


L1 net
from here
lidar median: 2986.00
rescaled median: 11.66
medians input rgb: 57.00 - 68.00 - 63.00
medians input rgb: 57.00 - 68.00, 63.00
(370, 1224, 3)
medians rescaled input rgb:  0.22 -  0.27 -  0.25
(1, 3, 370, 1224)
medians rescaled, transposed input rgb:  0.22 -  0.27 -  0.25
predicted median:  0.00
lidar quantiles:  6.13  -  17.57
predicted quantiles: -0.00  -   0.01




from eval
torch.Size([1, 3, 352, 1216])
medians rescaled, transposed input rgb:  0.32 -  0.33 -  0.29
lidar quantiles: 6  -  42
predicted median: 13
predicted quantiles: 6  -  72
rescaled predicted median: 13
rescaled predicted quantiles: 6  -  72



one_prediction from eval
lidar median: 3200.00
rescaled median: 12.50
medians input rgb: 81.00 - 83.00 - 75.00
medians input rgb: 81.00 - 83.00, 75.00
(352, 1216, 3)
medians rescaled input rgb:  0.32 -  0.33 -  0.29
(1, 3, 352, 1216)
medians rescaled, transposed input rgb:  0.32 -  0.33 -  0.29
lidar quantiles: 1616.00  -  10840.00
predicted median: 13.87
predicted quantiles:  6.38  -  72.03
rescaled predicted median: 13.87
rescaled predicted quantiles:  6.38  -  72.03


bis zu ca.2 % besser wenn testdaten zum early stopping verwendet werden würden
hier nichts in die richtung verwendet, da keins der netze es bisher tut
kein instanzbasierter loss



visualize 10 negativbeispiel
31 positiv? unterschiedliche ergebnisse bei mehrmaligem ausführen

