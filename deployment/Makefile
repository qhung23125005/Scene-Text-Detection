.PHONY: init deploy client clean streamlit

clean:
	ray stop || true
	rm -f tmp/ray_project/* || true

init:
	@if [ ! -d "tmp/ray_project" ]; then mkdir -p tmp/ray_project; fi
	ray start --head --disable-usage-stats --temp-dir="$(CURDIR)/tmp/ray_project" --dashboard-host=0.0.0.0

deploy:
	@echo "Starting object detection server..."
	serve run object_detection:entrypoint --address="auto"

deploy_ocr:
	@echo "Starting OCR server..."
	serve run ocr:entrypoint --address="auto"

streamlit:
	@echo "Starting Streamlit app..."
	streamlit run app.py