nohup /online/speech/peihao.wu/anaconda2/bin/python generator.py --config=../conf/single.conf --type=train --job_name=ps > log.ps &
nohup /online/speech/peihao.wu/anaconda2/bin/python generator.py --config=../conf/single.conf --type=train --job_name=worker --task_id=0 > log.0 &
