import json
import subprocess

def daphne(args, cwd='./daphne'):
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          capture_output=True, cwd=cwd)
    #print(proc.stdout)
    #print(proc.stderr)
    if(proc.returncode != 0):
        raise Exception(proc.stderr.decode())
    return json.loads(proc.stdout)