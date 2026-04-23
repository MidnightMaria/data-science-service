package com.agnesmaria.datascience.service;

import com.agnesmaria.datascience.dto.ModelEvaluationResponse;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.CSVReader;
import org.springframework.stereotype.Service;

import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class ModelEvaluationService {

    private static final String FINAL_SUMMARY_PATH =
            "reports/final_evaluation/test_summary.json";

    private static final String BASELINE_SUMMARY_PATH =
            "reports/baseline_comparison/baseline_summary.json";

    private static final String BEST_SERIES_PATH =
            "reports/plots/best_series.csv";

    private static final String WORST_SERIES_PATH =
            "reports/plots/worst_series.csv";

    private final ObjectMapper objectMapper;

    public ModelEvaluationService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public ModelEvaluationResponse getModelEvaluation() {
        try {
            Map<String, Object> finalSummary = readJson(FINAL_SUMMARY_PATH);
            Map<String, Object> baselineSummary = readJson(BASELINE_SUMMARY_PATH);

            ModelEvaluationResponse response = new ModelEvaluationResponse();
            response.setSummary(buildSummary(finalSummary, baselineSummary));
            response.setComparisonTable(buildComparisonTable(baselineSummary));
            response.setBestSeries(readSeriesCsv(BEST_SERIES_PATH));
            response.setWorstSeries(readSeriesCsv(WORST_SERIES_PATH));

            return response;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load model evaluation reports", e);
        }
    }

    private ModelEvaluationResponse.Summary buildSummary(
            Map<String, Object> finalSummary,
            Map<String, Object> baselineSummary
    ) {
        ModelEvaluationResponse.Summary summary = new ModelEvaluationResponse.Summary();

        Map<String, Object> globalProphet =
                castMap(finalSummary.get("global_prophet"));
        Map<String, Object> globalHybrid =
                castMap(finalSummary.get("global_hybrid"));
        Map<String, Object> hybridBaseline =
                castMap(baselineSummary.get("Hybrid"));

        summary.setProphetMae(toDouble(globalProphet.get("MAE")));
        summary.setProphetRmse(toDouble(globalProphet.get("RMSE")));
        summary.setProphetSmape(toDouble(globalProphet.get("SMAPE")));

        // Prefer baseline summary for hybrid global metrics table consistency
        summary.setHybridMae(toDouble(hybridBaseline.get("MAE")));
        summary.setHybridRmse(toDouble(hybridBaseline.get("RMSE")));
        summary.setHybridSmape(toDouble(hybridBaseline.get("SMAPE")));

        summary.setImprovedSeries(toInt(finalSummary.get("series_improved")));
        summary.setTotalSeries(toInt(finalSummary.get("total_series")));
        summary.setImprovementPct(toDouble(finalSummary.get("improvement_pct")));

        return summary;
    }

    private List<ModelEvaluationResponse.ModelMetric> buildComparisonTable(
            Map<String, Object> baselineSummary
    ) {
        List<ModelEvaluationResponse.ModelMetric> rows = new ArrayList<>();

        rows.add(buildMetricRow("Naive", castMap(baselineSummary.get("Naive"))));
        rows.add(buildMetricRow("Moving Average", castMap(baselineSummary.get("Moving_Average"))));
        rows.add(buildMetricRow("Prophet", castMap(baselineSummary.get("Prophet"))));
        rows.add(buildMetricRow("Hybrid", castMap(baselineSummary.get("Hybrid"))));

        return rows;
    }

    private ModelEvaluationResponse.ModelMetric buildMetricRow(
            String modelName,
            Map<String, Object> metrics
    ) {
        ModelEvaluationResponse.ModelMetric row = new ModelEvaluationResponse.ModelMetric();
        row.setModel(modelName);
        row.setMae(toDouble(metrics.get("MAE")));
        row.setRmse(toDouble(metrics.get("RMSE")));
        row.setMape(toDouble(metrics.get("MAPE")));
        row.setSmape(toDouble(metrics.get("SMAPE")));
        return row;
    }

    private List<ModelEvaluationResponse.SeriesMetric> readSeriesCsv(String path) throws Exception {
        Path filePath = Path.of(path);
        if (!Files.exists(filePath)) {
            throw new IllegalArgumentException("Report file not found: " + path);
        }

        List<ModelEvaluationResponse.SeriesMetric> results = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new FileReader(path))) {
            String[] headers = reader.readNext();
            if (headers == null) {
                return results;
            }

            String[] row;
            while ((row = reader.readNext()) != null) {
                ModelEvaluationResponse.SeriesMetric item =
                        new ModelEvaluationResponse.SeriesMetric();

                for (int i = 0; i < headers.length && i < row.length; i++) {
                    String header = headers[i].trim();
                    String value = row[i].trim();

                    switch (header) {
                        case "store" -> item.setStore(toInt(value));
                        case "item" -> item.setItem(toInt(value));
                        case "Prophet_SMAPE" -> item.setProphet(toDouble(value));
                        case "Hybrid_SMAPE" -> item.setHybrid(toDouble(value));
                        case "SMAPE_Improvement" -> item.setImprovement(toDouble(value));
                    }
                }

                results.add(item);
            }
        }

        return results;
    }

    private Map<String, Object> readJson(String filePath) throws Exception {
        Path path = Path.of(filePath);

        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Report file not found: " + filePath);
        }

        return objectMapper.readValue(
                Files.readString(path),
                new TypeReference<Map<String, Object>>() {}
        );
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> castMap(Object value) {
        return (Map<String, Object>) value;
    }

    private int toInt(Object value) {
        if (value == null) return 0;
        if (value instanceof Number number) {
            return number.intValue();
        }
        return Integer.parseInt(value.toString());
    }

    private double toDouble(Object value) {
        if (value == null) return 0.0;
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        return Double.parseDouble(value.toString());
    }
}