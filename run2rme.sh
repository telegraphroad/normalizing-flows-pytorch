flows="defaultglow.yaml"
losses="TA  ML"
ddistr="ggd"
bases="mvggd mvn"


for dim in 2
do
	for flow in $flows
	do 
		for ddist in $ddistr
		do
			for fbeta in -2. -1. -0.5 -0.2 -0.1 -0.01
			do
				rm configs/default.yaml
				cp configs/"$flow" configs/default.yaml
				varn="True"
				for dbeta in 0.4 1.2 2.0 2.8 3.6
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
													python main651_rme.py +run.nloss_type="$nlt" +run.bloss_type="$blt" +run.vprior="$b" +run.vvariable="$v" +run.vnbeta="$nbeta" +run.vdbeta="$dbeta" +run.ddim="$dim" +run.niters=1 +run.biters=1 +run.ddistrib="$ddist" +run.fbeta="$fbeta"
												done
										done
								done
						done
						
					done
				done

				varn="False"
				for dbeta in 0.4 1.2 2.0 2.8 3.6
				do
					for nbeta in 2.0
					do
						for nlt in $losses
						do
							for b in $bases
								do
									for v in $varn
										do
											python main651_rme.py +run.nloss_type="$nlt" +run.bloss_type="$nlt" +run.vprior="$b" +run.vvariable="$v" +run.vnbeta="$nbeta" +run.vdbeta="$dbeta" +run.ddim="$dim" +run.niters=1 +run.biters=1 +run.ddistrib="$ddist"  +run.fbeta="$fbeta"
										done
								done
						done
						
					done
				done
				#mv logs "/PhD/Synth/normalizing-flows-pytorch/logs11_dim_$dim_${flow}_${ddist}_fbeta_$fbeta"
			done
                        
		done
	done
done
