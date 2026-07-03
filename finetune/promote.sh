#!/bin/bash
# Промоут обученного адаптера в production как версию.
#
# Использование:
#   bash finetune/promote.sh <adapter-path> <version> "описание"
# Пример:
#   bash finetune/promote.sh finetune/adapters_persona v1-persona "Персона Тасочки, Gemma 3 4B"
#
# База по умолчанию — та же Gemma 4B; переопределить: BASE=... bash finetune/promote.sh ...
# Standalone-экспорт (слить адаптер в самодостаточную модель): FUSE=1 bash finetune/promote.sh ...
set -e
cd "$(dirname "$0")/.."
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"

ADAPTER="$1"; VERSION="$2"; DESC="${3:-}"
BASE="${BASE:-mlx-community/gemma-3-4b-it-qat-4bit}"

if [ -z "$ADAPTER" ] || [ -z "$VERSION" ]; then
  echo "usage: bash finetune/promote.sh <adapter-path> <version> [описание]"
  exit 1
fi
if [ ! -d "$ADAPTER" ]; then
  echo "Адаптер не найден: $ADAPTER"; exit 1
fi

DEST="production/$VERSION"
mkdir -p "$DEST/adapters"
# Только финальный адаптер + конфиг (без промежуточных чекпоинтов 0000NNN_*).
cp "$ADAPTER"/adapters.safetensors "$DEST/adapters/" 2>/dev/null || true
cp "$ADAPTER"/adapter_config.json "$DEST/adapters/" 2>/dev/null || true

cat > "$DEST/config.yaml" <<EOF
version: $VERSION
base_model: $BASE
adapter: adapters/
created: $(date +%Y-%m-%d)
description: $DESC
EOF

cat > "$DEST/run.sh" <<'RUN'
#!/bin/bash
# Чат с этой версией модели. Пример: bash run.sh "Как тебя зовут?"
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.tasochka-finetune-venv/bin/python"
BASE=$(awk '/^base_model:/{print $2}' "$DIR/config.yaml")
$PYTHON -m mlx_lm generate --model "$BASE" --adapter-path "$DIR/adapters" \
  --prompt "${1:-Как тебя зовут?}" --max-tokens 300 --temp 0.7
RUN
chmod +x "$DEST/run.sh"

if [ ! -f "$DEST/model_card.md" ]; then
  cat > "$DEST/model_card.md" <<EOF
# $VERSION

$DESC

- **База:** \`$BASE\`
- **Тип:** LoRA-адаптер (запускается поверх базы)
- **Создано:** $(date +%Y-%m-%d)

## Запуск
\`\`\`bash
bash production/$VERSION/run.sh "Привет, как тебя зовут?"
\`\`\`

## Оценка
(заполни после quiz.py: персона X/30, протечки Y/30, качество Java)
EOF
fi

if [ "${FUSE:-0}" = "1" ]; then
  echo "Слияние в самодостаточную модель (FUSE=1)..."
  $PYTHON -m mlx_lm fuse --model "$BASE" --adapter-path "$ADAPTER" --save-path "$DEST/fused"
  echo "Standalone-модель: $DEST/fused"
fi

echo "Промоут готов: $DEST"
ls -la "$DEST"