"""Script for downloading all GLUE data.

Adapted from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e

"""
import argparse
import os
import sys
import urllib.request
import zipfile

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI",
         "RTE", "WNLI", "diagnostic"]

# pylint: disable=line-too-long

TASK2PATH = {
    "CoLA": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4',
    "SST": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
    "MRPC": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc',
    "QQP": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5',
    "STS": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5',
    "MNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce',
    "SNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df',
    "QNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601',
    "RTE": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb',
    "WNLI": 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf',
    "diagnostic": 'https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D'}

# pylint: enable=line-too-long


def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


def format_mrpc(data_dir, path_to_data):
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
    assert os.path.isfile(mrpc_train_file), \
        "Train data not found at %s" % mrpc_train_file
    assert os.path.isfile(mrpc_test_file), \
        "Test data not found at %s" % mrpc_test_file
    urllib.request.urlretrieve(TASK2PATH["MRPC"],
                               os.path.join(mrpc_dir, "dev_ids.tsv"))

    dev_ids = []
    with open(os.path.join(mrpc_dir, "dev_ids.tsv")) as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file) as data_fh, \
            open(os.path.join(mrpc_dir, "train.tsv"), 'w') as train_fh, \
            open(os.path.join(mrpc_dir, "dev.tsv"), 'w') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write(
                    "%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
    with open(mrpc_test_file) as data_fh, \
            open(os.path.join(mrpc_dir, "test.tsv"), 'w') as test_fh:
        _ = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
    print("\tCompleted!")


def download_diagnostic(data_dir):
    print("Downloading and extracting diagnostic...")
    if not os.path.isdir(os.path.join(data_dir, "diagnostic")):
        os.mkdir(os.path.join(data_dir, "diagnostic"))
    data_file = os.path.join(data_dir, "diagnostic", "diagnostic.tsv")
    urllib.request.urlretrieve(TASK2PATH["diagnostic"], data_file)
    print("\tCompleted!")
    return


def get_tasks(task_names):
    task_names = task_names.split(',')
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', help='directory to save data to',
        type=str, default='data')
    parser.add_argument(
        '--tasks',
        help='tasks to download data for as a comma separated string',
        type=str, default='all')
    parser.add_argument(
        '--path_to_mrpc',
        help='path to directory containing extracted MRPC data, '
             'msr_paraphrase_train.txt and msr_paraphrase_text.txt',
        type=str, default='')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        if task == 'MRPC':
            import subprocess
            if not os.path.exists("data/MRPC"):
                subprocess.run("mkdir data/MRPC", shell=True)
            # pylint: disable=line-too-long
            subprocess.run(
                'wget -P data/MRPC/ https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt',
                shell=True)
            subprocess.run(
                'wget -P data/MRPC/ https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt',
                shell=True)
            # pylint: enable=line-too-long
            format_mrpc(args.data_dir, args.path_to_mrpc)
            subprocess.run('rm data/MRPC/msr_paraphrase_train.txt', shell=True)
            subprocess.run('rm data/MRPC/msr_paraphrase_test.txt', shell=True)
        elif task == 'diagnostic':
            download_diagnostic(args.data_dir)
        else:
            download_and_extract(task, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
