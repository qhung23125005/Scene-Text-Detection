.PHONY: init deploy deploy_ocr client clean streamlit

clean:
	@if exist tmp\ray_project ( \
		ray stop || exit /b 0 \
	) && \
	del /q tmp\ray_project\* 2>nul || exit /b 0

init:
	@if not exist tmp\ray_project mkdir tmp\ray_project
	@ray start --head --disable-usage-stats --temp-dir="$(CURDIR)\tmp\ray_project" --dashboard-host=0.0.0.0 

deploy_ocr:
	@echo Starting OCR server...
	@serve run ocr:entrypoint --address="auto"

streamlit:
	@echo Starting Streamlit app...
	@streamlit run app.py
