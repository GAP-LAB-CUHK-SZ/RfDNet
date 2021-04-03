import sys
sys.path.append('.')
import os
import argparse
import ntpath
from glob import glob
from multiprocessing import Pool
import trimesh
import shutil

class Simplification:
    """
    Constructor.
    """
    def __init__(self):
        parser = self.get_parser()
        self.options = parser.parse_args()
        self.simplification_script = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'simplification.mlx')

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """

        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--in_dir', type=str,
                                 help='Path to input directory.')
        input_group.add_argument('--in_file', type=str,
                                 help='Path to input directory.')
        parser.add_argument('--out_dir', type=str,
                            help='Path to output directory; files within are overwritten!')

        return parser

    def read_directory(self, directory):
        """
        Read directory.

        :param directory: path to directory
        :return: list of files
        """

        files = []
        for filename in os.listdir(directory):
            files += glob(os.path.join(directory, filename, '*.off'))

        return files

    def get_in_files(self):
        if self.options.in_dir is not None:
            assert os.path.exists(self.options.in_dir)
            files = self.read_directory(self.options.in_dir)
        else:
            files = [self.options.in_file]

        return files

    def simplify(self, filepath):
        clsname = filepath.split('/')[3]
        output_file = os.path.join(self.options.out_dir, clsname, ntpath.basename(filepath))
        if not os.path.exists(output_file):
            if not os.path.exists(os.path.dirname(output_file)):
                os.mkdir(os.path.dirname(output_file))
            os.system('meshlabserver -i %s -o %s -s %s' % (
                filepath,
                output_file,
                self.simplification_script
            ))

        mesh = trimesh.load(output_file, process=False)
        if not mesh.is_watertight:
            with open(os.path.join(self.options.out_dir, 'not_watertight_list.txt'), 'a+') as file:
                file.write(output_file + '\n')

            os.remove(output_file)
            shutil.copy2(filepath, output_file)

    def run(self):
        """
        Run simplification.
        """
        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)
        files = self.get_in_files()

        p = Pool(processes=16)
        p.map(self.simplify, files)
        p.close()
        p.join()


if __name__ == '__main__':
    app = Simplification()
    app.run()