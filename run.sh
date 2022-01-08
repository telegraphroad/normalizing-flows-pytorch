flows="defaultglow.yaml"
losses="TA  ML"
ddistr="credit"
bases="mvn"


for dim in 30
do


	for flow in $flows
	do 
		for ddist in $ddistr
		do

			rm logs

			rm configs/default.yaml
			cp configs/"$flow" configs/default.yaml




			varn="True"

			for dbeta in 1
			do
				for nbeta in 5.0
				do
					for nlt in $losses
					do
						for b in $bases
							do
								for v in $varn
									do
										for blt in $losses
											do
												python main650.py +run.nloss_type="$nlt" +run.bloss_type="$blt" +run.vprior="$b" +run.vvariable="$v" +run.vnbeta="$nbeta" +run.vdbeta="$dbeta" +run.ddim="$dim" +run.niters=1 +run.biters=1 +run.ddistrib="$ddist"
											done
									done
							done
					done
					
				done
			done

			varn="False"
			for dbeta in 1
			do
				for nbeta in 2.0
				do
					for nlt in $losses
					do
						for b in $bases
							do
								for v in $varn
									do
										python main650.py +run.nloss_type="$nlt" +run.bloss_type="$nlt" +run.vprior="$b" +run.vvariable="$v" +run.vnbeta="$nbeta" +run.vdbeta="$dbeta" +run.ddim="$dim" +run.niters=1 +run.biters=1 +run.ddistrib="$ddist"
									done
							done
					done
					
				done
			done
			mv logs "logs11$dim${flow}${ddist}"
		done
	
	done
done











