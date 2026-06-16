.PHONY: clean

clean:
	@echo "Limpando artefatos de build do Rust..."
	rm -rf target/
	rm -rf tauri-chat-lfm/src-tauri/target/
	@echo "Limpando artefatos de build e dependências do Node.js..."
	rm -rf node_modules/
	rm -rf tauri-chat-lfm/node_modules/
	rm -rf tauri-chat-lfm/dist/
	rm -rf tauri-chat-lfm/src-tauri/gen/
	@echo "Limpeza concluída com sucesso."
