"""Combines prof data from multiple runs into a single file."""
import glob, os, pstats


def main():
    """Main function."""
    prof_dir = os.path.join(os.path.dirname(__file__), "prof")
    files = glob.glob(os.path.join(prof_dir, "prof*.prof"))
    stats = pstats.Stats(files[0])
    for f in files[1:]:
        stats.add(f)
    stats.strip_dirs().sort_stats('cumulative')
    
    i = 0
    combined_name = f'combined_{len(files)}_{i}.prof'
    while os.path.exists(os.path.join(prof_dir, combined_name)):
        i += 1
        combined_name = f'combined_{len(files)}_{i}.prof'
    stats.dump_stats(os.path.join(prof_dir, combined_name))
    for f in files:
        os.remove(f)

if __name__ == '__main__':
    main()