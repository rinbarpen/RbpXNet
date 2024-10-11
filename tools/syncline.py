import subprocess

TASKS = [
    [
        "configs/Segment-Drive/UNet_DRIVE.json",
        '--train'
    ],
]

for task in TASKS:
    cmd = []
    for x in task:
        if x == "config":
            cmd.extend(["python", "main.py", "-c", x])
        else:
            cmd.append(x)
    try:
        subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Failed to execute with error: {e}')
        raise e
