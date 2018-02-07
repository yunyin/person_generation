nohup /online/speech/peihao.wu/anaconda2/bin/python generator.py --config=../conf/train.conf --type=train --job_name=ps > log.ps &
nohup /online/speech/peihao.wu/anaconda2/bin/python generator.py --config=../conf/train.conf --type=train --job_name=worker --task_id=0 > log.0 &
nohup /online/speech/peihao.wu/anaconda2/bin/python generator.py --config=../conf/train.conf --type=train --job_name=worker --task_id=1 > log.1 &
