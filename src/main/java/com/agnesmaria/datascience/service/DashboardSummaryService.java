package com.agnesmaria.datascience.service;

import com.agnesmaria.datascience.dto.DashboardSummaryResponse;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

@Service
public class DashboardSummaryService {

    private static final String FINAL_SUMMARY_PATH =
            "reports/final_evaluation/test_summary.json";

    private static final String INVENTORY_SUMMARY_PATH =
            "reports/inventory_optimization/inventory_summary.json";

    private final ObjectMapper objectMapper;

    public DashboardSummaryService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    public DashboardSummaryResponse getDashboardSummary() {
        try {
            Map<String, Object> finalSummary = readJson(FINAL_SUMMARY_PATH);
            Map<String, Object> inventorySummary = readJson(INVENTORY_SUMMARY_PATH);

            DashboardSummaryResponse response = new DashboardSummaryResponse();

            // =========================
            // FINAL EVALUATION SUMMARY
            // =========================
            response.setTotalSeries(toInt(finalSummary.get("total_series")));
            response.setImprovedSeries(toInt(finalSummary.get("series_improved")));
            response.setImprovementPct(toDouble(finalSummary.get("improvement_pct")));

            // =========================
            // INVENTORY SUMMARY
            // =========================
            response.setReorder(toInt(inventorySummary.get("n_reorder")));
            response.setSafe(toInt(inventorySummary.get("n_safe")));

            // Tidak semua report punya overstock
            response.setOverstock(toInt(inventorySummary.getOrDefault("n_overstock", 0)));

            response.setAvgSafetyStock(toDouble(inventorySummary.get("avg_safety_stock")));
            response.setAvgReorderPoint(toDouble(inventorySummary.get("avg_reorder_point")));
            response.setAvgEoq(toDouble(inventorySummary.get("avg_eoq")));

            return response;

        } catch (Exception e) {
            throw new RuntimeException("Failed to load dashboard summary reports", e);
        }
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