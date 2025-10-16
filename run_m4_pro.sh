#!/bin/bash
# Automated ML Training for MacBook M4 Pro
# Run this script to execute the complete training pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Khmer PDF ML Training - M4 Pro${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if we're in the right directory
if [ ! -d "data/pdfs_all" ]; then
    echo -e "${RED}Error: Run this script from ml_correction/ directory${NC}"
    exit 1
fi

# Step 1: Check dependencies
echo -e "${YELLOW}Step 1: Checking dependencies...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check Tesseract
if ! command -v tesseract &> /dev/null; then
    echo -e "${YELLOW}Tesseract not found. Installing via Homebrew...${NC}"
    brew install tesseract tesseract-lang
fi

if ! tesseract --list-langs 2>&1 | grep -q "khm"; then
    echo -e "${RED}Khmer language data not found${NC}"
    echo -e "${YELLOW}Installing tesseract-lang...${NC}"
    brew install tesseract-lang
fi
echo -e "${GREEN}✓ Tesseract with Khmer support ready${NC}"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Installing PyTorch for Apple Silicon...${NC}"
    pip3 install torch torchvision torchaudio
fi

# Check MPS availability
if ! python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    echo -e "${YELLOW}⚠ MPS not available, will use CPU (slower)${NC}"
    DEVICE="cpu"
else
    echo -e "${GREEN}✓ MPS (Apple Silicon GPU) available${NC}"
    DEVICE="mps"
fi

# Check other dependencies
echo -e "${YELLOW}Checking Python dependencies...${NC}"
pip3 install -q -r requirements.txt 2>/dev/null || echo -e "${YELLOW}Some dependencies may need manual installation${NC}"

# Step 2: Ask user what to do
echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}What would you like to do?${NC}"
echo -e "${YELLOW}========================================${NC}"
echo "1) Quick test (50 pages, 1 hour) - RECOMMENDED FIRST"
echo "2) Production training (500 pages, overnight)"
echo "3) Full scale training (2000 pages, 1-2 nights)"
echo "4) Just collect data (no training)"
echo "5) Test existing model"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        # Quick test
        echo -e "\n${GREEN}Starting quick test (50 pages)...${NC}"

        # Collect data
        echo -e "${YELLOW}Collecting 50-page dataset...${NC}"
        python data/collect_training_pairs.py \
            --pdf-dir data/pdfs_all \
            --output-dir data/training_pairs_small \
            --num-pages 50 \
            --pages-per-pdf 1 \
            --ocr-lang khm \
            --min-khmer-ratio 0.3

        # Train LSTM
        echo -e "${YELLOW}Training LSTM baseline...${NC}"
        python training/train.py \
            --model lstm \
            --data-dir data/training_pairs_small \
            --output-dir checkpoints/lstm_m4_50p \
            --epochs 50 \
            --batch-size 64 \
            --lr 0.001 \
            --device $DEVICE

        # Show results
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}Quick test complete!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo "Results saved to: checkpoints/lstm_m4_50p/"
        echo "Check final CER:"
        tail -5 checkpoints/lstm_m4_50p/training.log
        ;;

    2)
        # Production training (500 pages)
        echo -e "\n${GREEN}Starting production training (500 pages)...${NC}"
        echo -e "${YELLOW}This will run overnight. Use 'tail -f' to monitor progress.${NC}"

        # Collect data
        echo -e "${YELLOW}Step 1/2: Collecting 500-page dataset...${NC}"
        nohup python data/collect_training_pairs.py \
            --pdf-dir data/pdfs_all \
            --output-dir data/training_pairs_500 \
            --num-pages 500 \
            --ocr-lang khm \
            --min-khmer-ratio 0.3 \
            > collect_500.log 2>&1 &

        COLLECT_PID=$!
        echo "Data collection started (PID: $COLLECT_PID)"
        echo "Monitor with: tail -f collect_500.log"
        echo ""
        echo "When collection completes, train with:"
        echo "nohup python training/train.py --model transformer --data-dir data/training_pairs_500 --output-dir checkpoints/transformer_m4_500 --epochs 100 --batch-size 32 --mixed-precision --device $DEVICE > train_500.log 2>&1 &"
        ;;

    3)
        # Full scale (2000 pages)
        echo -e "\n${GREEN}Starting full-scale training (2000 pages)...${NC}"
        echo -e "${YELLOW}This will take 1-2 nights. Recommended to run overnight.${NC}"

        # Collect data
        echo -e "${YELLOW}Step 1/2: Collecting 2000-page dataset...${NC}"
        nohup python data/collect_training_pairs.py \
            --pdf-dir data/pdfs_all \
            --output-dir data/training_pairs_full \
            --num-pages 2000 \
            --ocr-lang khm \
            --min-khmer-ratio 0.3 \
            > collect_full.log 2>&1 &

        COLLECT_PID=$!
        echo "Data collection started (PID: $COLLECT_PID)"
        echo "Monitor with: tail -f collect_full.log"
        echo ""
        echo "When collection completes, train with:"
        echo "nohup python training/train.py --model transformer --data-dir data/training_pairs_full --output-dir checkpoints/transformer_m4_full --epochs 100 --batch-size 32 --mixed-precision --device $DEVICE > train_full.log 2>&1 &"
        ;;

    4)
        # Just collect data
        echo -e "\n${YELLOW}How many pages to collect?${NC}"
        read -p "Enter number (e.g., 100): " num_pages

        echo -e "${YELLOW}Collecting ${num_pages}-page dataset...${NC}"
        python data/collect_training_pairs.py \
            --pdf-dir data/pdfs_all \
            --output-dir data/training_pairs_${num_pages} \
            --num-pages $num_pages \
            --ocr-lang khm \
            --min-khmer-ratio 0.3

        echo -e "${GREEN}Dataset saved to: data/training_pairs_${num_pages}/${NC}"
        ;;

    5)
        # Test existing model
        echo -e "\n${YELLOW}Available models:${NC}"
        find checkpoints -name "best_model.pt" -type f 2>/dev/null || echo "No models found"
        echo ""
        read -p "Enter model path: " model_path

        if [ ! -f "$model_path" ]; then
            echo -e "${RED}Model not found: $model_path${NC}"
            exit 1
        fi

        echo -e "${YELLOW}Testing model: $model_path${NC}"
        python << EOF
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from inference.corrector import KhmerCorrector

corrector = KhmerCorrector('$model_path', device='$DEVICE', use_beam_search=False)

# Test texts
tests = [
    "ក្ងនការដោះស្រាយ",
    "ភាសាខ្មែរ",
]

print("\n" + "="*50)
print("Model Testing Results")
print("="*50)

for text in tests:
    corrected = corrector.correct_text(text)
    print(f"\nInput:     {text}")
    print(f"Corrected: {corrected}")

print("\n" + "="*50)
EOF
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Script complete!${NC}"
echo -e "${GREEN}========================================${NC}"
