cd ../../

if [ -f "./scripts/env/env.sh" ]; then
    source ./scripts/env/env.sh
fi

python main.py dataset=slimpajama_120b dataset.component_whitelist=null "$@"