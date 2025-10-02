cd ../../

if [ -f "./scripts/env/env.sh" ]; then
    source ./scripts/env/env.sh
fi

python main.py dataset=slimpajama_5m dataset.component_whitelist=[RedPajamaCommonCrawl] "$@"