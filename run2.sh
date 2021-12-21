flows="defaultglow.yaml"

for dim in 3 10 5 2
do

	for flow in $flows
	do 
		rm logs

		rm configs/default.yaml
		cp configs/"$flow" configs/default.yaml

		losses="TA  ML"
		bases="mvn  ggd"
		varn="False"
		for dbeta in 0.4 1.2 2.0 2,8 3.6
		do
			for nbeta in 2.0
			do
				for lt in $losses
				do
					for b in $bases
						do
							for v in $varn
								do
									python main651.py +run.loss_type="$lt" +run.vprior="$b" +run.vvariable="$v" +run.vnbeta="$nbeta" +run.vdbeta="$dbeta" +run.ddim="$dim"
								done
						done
				done
				
			done
		done



		losses="TA  ML"
		bases="mvn  ggd"
		varn="True"
		for dbeta in 0.4 1.2 2.0 2,8 3.6
		do
			for nbeta in 5.0
			do
				for lt in $losses
				do
					for b in $bases
						do
							for v in $varn
								do
									python main651.py +run.loss_type="$lt" +run.vprior="$b" +run.vvariable="$v" +run.vnbeta="$nbeta" +run.vdbeta="$dbeta" +run.ddim="$dim"
								done
						done
				done
				
			done
		done


		mv logs "logs$dim"
	done
done
