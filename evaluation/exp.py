#!/usr/bin/env python3

import os
import glob
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt


def prepare(args):
    '''Generates input files used to evaluate the HyperLogLog algorithms.

    Can generate either files containing random 32/64 bit hashes or a file
    with cardinalities used to measure cardinality estimations in the second
    group of experiments.

    '''
    if args.type == 'hashes':
        command = [
            args.exe,
            '--jobs', str(args.jobs),
            'gen',
            '--size', args.size,
            '--count', str(args.count),
            '--runs', str(args.runs),
            '--output', args.output
        ]

        subprocess.run(command, check=True)
    else:
        filepath = os.path.join(args.output, 'cardinalities.dat')

        cardinalities = generate_cardinalities(args.max_cardinality)

        with open(filepath, 'w') as out:
            out.write('\n'.join(map(str, cardinalities)))


def run(args):
    '''Runs the evaluation program for each of the input files.'''

    command = [
        args.exe,
        '--jobs', str(args.jobs),
        'run',
        '--type', args.type,
        '--mode', args.mode,
        '--precision', str(args.precision),
        '--output', args.output
    ]

    if args.mode == 'cardinalities':
        command.extend([
            '--runs', str(args.runs),
            '--cardinalities', args.cardinalities,
        ])
    else:
        command.append('--input')
        command.extend(glob.glob('{}/*.dat'.format(args.input)))

    subprocess.run(command, check=True)


def plot(args):
    '''Generates different types of plots based on the results.'''

    data = calculate_statistics(args.input)

    max_cardinality = int(args.max_cardinality)

    if args.type == 'hll':
        impl = 'HyperLogLog'

        if args.mode == 'hashes':
            ylim, ytick_step = [0, 0.035], 0.0025
        else:
            ylim, ytick_step = [-0.01, 0.135], 0.010
    else:
        impl = 'HyperLogLog++'

        if args.mode == 'hashes':
            ylim, ytick_step = [0, 0.005], 0.0005
        else:
            ylim, ytick_step = [-0.00025, 0.006], 0.0005

    cardinalities, means, medians, stdevs = data

    means = np.array(means)

    stdevs = np.array(stdevs)

    _, ax = plt.subplots()

    median_line = ax.plot(cardinalities, medians, '-', linewidth=0.5)

    ax.fill_between(cardinalities, means - stdevs, means + stdevs, alpha=0.2)

    mean_line = ax.plot(cardinalities, means, '-', linewidth=0.5)

    plt.yticks(np.arange(0, max(means) + max(stdevs), ytick_step))

    if args.mode == 'hashes':
        ax.set_xlim(0, max_cardinality)
    else:
        ax.set_xscale('log')
        ax.set_xlim(right=max_cardinality)

    ax.set_ylim(*ylim)

    plt.grid(linestyle='--')

    plt.xlabel('Cardinality')
    plt.ylabel('Relative Error')

    plt.title('{} Accuracy (precision: {})'.format(impl, args.precision))

    if args.mode == 'hashes':
        plt.legend((mean_line[0], median_line[0]), ('mean', 'median'))
    else:
        plt.legend(
            (mean_line[0], median_line[0]),
            ('mean', 'median'),
            loc='upper left'
        )

    plt.show()


def calculate_statistics(location):
    '''Calculates statistics from estimations.'''
    files = [
        open(filename, 'r')
        for filename in glob.glob('{}/*.dat'.format(location))
    ]

    cardinalities, means, medians, stdevs = [], [], [], []

    while True:
        lines = []

        for fp in files:
            line = fp.readline()

            if line:
                lines.append(line)

        if not lines:
            break

        line_values = [list(map(float, values.split(' '))) for values in lines]

        relative_error = []
        for line in line_values:
            relative_error.append(abs(line[1] - line[0]) / line[0])

        cardinalities.append(int(line_values[0][0]))
        means.append(np.mean(relative_error))
        medians.append(np.median(relative_error))
        stdevs.append(np.std(relative_error))

    for fp in files:
        fp.close()

    return (cardinalities, means, medians, stdevs)


def generate_cardinalities(max_cardinality, ratio=1.007):
    '''Returns a sorted list of numbers following a geometric series.'''
    cur, data = 1, set()

    while cur < max_cardinality:
        data.add(int(cur))

        cur *= ratio

    data.add(int(cur))

    return sorted(data)


def parse_args():
    '''Parses command line arguments.'''
    parser = argparse.ArgumentParser(prog='exp')

    parser.add_argument(
        '-x',
        '--exe',
        type=str,
        default='./target/release/evl',
        help='specify exe location')

    parser.add_argument(
        '-j',
        '--jobs',
        default=1,
        type=int,
        help='specify number of jobs to spawn')

    subparsers = parser.add_subparsers(dest='subcommand')

    prepare_parser = subparsers.add_parser('prepare')

    prepare_parser.add_argument(
        '-t',
        '--type',
        required=True,
        choices=['hashes', 'cardinalities'],
        help='specify execution mode')

    prepare_parser.add_argument(
        '-s',
        '--size',
        default='32',
        choices=['32', '64'],
        help='specify the hash size in bits')

    prepare_parser.add_argument(
        '-c',
        '--count',
        type=int,
        default=100000,
        help='specify number of hashes per run/file')

    prepare_parser.add_argument(
        '-r',
        '--runs',
        type=int,
        default=100,
        help='specify number of runs/files')

    prepare_parser.add_argument(
        '-a',
        '--max-cardinality',
        type=int,
        default=1000000000,
        help='specify max cardinality')

    prepare_parser.add_argument(
        'output',
        type=str,
        help='specify output location')

    run_parser = subparsers.add_parser('run')

    run_parser.add_argument(
        '-t',
        '--type',
        required=True,
        default='hll',
        choices=['hll', 'hllp'],
        help='specify hyperloglog implementation')

    run_parser.add_argument(
        '-m',
        '--mode',
        required=True,
        default='hashes',
        choices=['hashes', 'cardinalities'],
        help='specify execution mode')

    run_parser.add_argument(
        '-p',
        '--precision',
        type=int,
        default=14,
        help='specify hyperloglog precision')

    run_parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='specify input location')

    run_parser.add_argument(
        '-r',
        '--runs',
        type=int,
        help='specify runs')

    run_parser.add_argument(
        '-c',
        '--cardinalities',
        help='specify cardinalities file')

    run_parser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='specify output location')

    plot_parser = subparsers.add_parser('plot')

    plot_parser.add_argument(
        '-t',
        '--type',
        required=True,
        default='hll',
        choices=['hll', 'hllp'],
        help='specify hyperloglog implementation')

    plot_parser.add_argument(
        '-m',
        '--mode',
        required=True,
        default='hashes',
        choices=['hashes', 'cardinalities'],
        help='specify execution mode')

    plot_parser.add_argument(
        '-p',
        '--precision',
        type=int,
        default=14,
        help='specify hyperloglog precision')

    plot_parser.add_argument(
        '-a',
        '--max-cardinality',
        type=int,
        default=100000,
        help='specify max cardinality')

    plot_parser.add_argument(
        'input',
        type=str,
        help='specify input location')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()

    if arguments.subcommand == 'prepare':
        prepare(arguments)
    elif arguments.subcommand == 'run':
        run(arguments)
    elif arguments.subcommand == 'plot':
        plot(arguments)
