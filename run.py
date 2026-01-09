"""
CondMol
Select script to execute via --mode parameter, other parameters are passed through normally
"""

import sys
import os
import argparse
import subprocess
import importlib.util


def main():
    parser = argparse.ArgumentParser(
        description='CondMol Unified Entry Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available mode options:
  taskemb      - Generate task embeddings (taskemb.py)
  condemb      - Generate condition embeddings (condemb.py)
  preprocess   - Preprocess data (preprocess.py)
  finetune     - Fine-tune model (run_finetune.py)
  zeroshot     - Zero-shot prediction (run_zeroshot.py)
  condact      - CondACT experiment (run_condact.py)
  condactfew   - CondACT_few experiment (run_condactfew.py)
  condadme     - CondADME experiment (run_condadme.py)

Examples:
  python run.py --mode taskemb --moldata CHEMBL218 --target AKT1 --keyword inhibitor --task_file tasks.npy
  python run.py --mode finetune --moldata CHEMBL218 --pretrain checkpoints/model_CSLoss.pkl --pi None
        """
    )
    
    # mode parameter, required
    parser.add_argument('--mode', type=str, required=True,
                       choices=['taskemb', 'condemb', 'preprocess', 'finetune', 'zeroshot', 
                               'condact', 'condactfew', 'condadme'],
                       help='Select script mode to execute')
    
    # Parse mode parameter
    args, unknown = parser.parse_known_args()
    
    # Mode to script file mapping
    mode_to_script = {
        'taskemb': 'scripts/taskemb.py',
        'condemb': 'scripts/condemb.py',
        'preprocess': 'scripts/preprocess.py',
        'finetune': 'scripts/run_finetune.py',
        'zeroshot': 'scripts/run_zeroshot.py',
        'condact': 'scripts/run_condact.py',
        'condactfew': 'scripts/run_condactfew.py',
        'condadme': 'scripts/run_condadme.py',
    }
    
    # Get corresponding script file path (use absolute path)
    script_path = os.path.abspath(mode_to_script[args.mode])
    
    # Check if file exists
    if not os.path.exists(script_path):
        print(f"Error: Script file does not exist: {script_path}", file=sys.stderr)
        sys.exit(1)
    
    # 方法1: 直接使用subprocess调用（推荐，完全隔离环境）
    # 构建新的命令行参数
    cmd_args = ["python", script_path]
    
    # 添加所有其他参数（除了--mode）
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--mode':
            # Skip --mode and its value
            if i + 1 < len(sys.argv):
                i += 2
            else:
                i += 1
        elif arg.startswith('--mode='):
            # Handle --mode=value format, skip directly
            i += 1
        else:
            cmd_args.append(arg)
            i += 1
    
    try:
        # 使用子进程执行，这样可以完全隔离环境
        result = subprocess.run(cmd_args, check=True)
        sys.exit(result.returncode)
        
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed with return code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error executing script {script_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()