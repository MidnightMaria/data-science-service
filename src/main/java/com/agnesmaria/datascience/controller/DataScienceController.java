package com.agnesmaria.datascience.controller;

import com.agnesmaria.datascience.dto.DashboardSummaryResponse;
import com.agnesmaria.datascience.service.CSVReaderService;
import com.agnesmaria.datascience.service.DashboardSummaryService;
import org.springframework.web.bind.annotation.*;
import com.agnesmaria.datascience.dto.ModelEvaluationResponse;
import com.agnesmaria.datascience.service.ModelEvaluationService;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/data-science")
public class DataScienceController {

    private final CSVReaderService csvReaderService;
    private final DashboardSummaryService dashboardSummaryService;
    private final ModelEvaluationService modelEvaluationService;

    public DataScienceController(
        CSVReaderService csvReaderService,
        DashboardSummaryService dashboardSummaryService,
        ModelEvaluationService modelEvaluationService
    ) {
        this.csvReaderService = csvReaderService;
        this.dashboardSummaryService = dashboardSummaryService;
        this.modelEvaluationService = modelEvaluationService;
    }

    @GetMapping("/inventory-optimization")
    public List<Map<String, String>> inventoryOptimization() {
        return csvReaderService.readCSV(
                "reports/inventory_optimization/inventory_policy_report.csv"
        );
    }

    @GetMapping("/demand-forecast")
    public List<Map<String, String>> demandForecast() {
        return csvReaderService.readCSV(
                "reports/forecast/future_demand_forecast.csv"
        );
    }

    @GetMapping("/dashboard-summary")
    public DashboardSummaryResponse dashboardSummary() {
        return dashboardSummaryService.getDashboardSummary();
    }
    
    @GetMapping("/model-evaluation")
    public ModelEvaluationResponse modelEvaluation() {
        return modelEvaluationService.getModelEvaluation();
    }
    /**
     * Endpoint lama tetap dipertahankan kalau masih dipakai tempat lain.
     * Tapi sekarang sudah lebih aman karena cek dua kemungkinan field:
     * inventory_status atau status
     */
    @GetMapping("/dashboard")
    public Map<String, Integer> dashboard() {

        List<Map<String, String>> data =
                csvReaderService.readCSV(
                        "reports/inventory_optimization/inventory_policy_report.csv"
                );

        int reorder = 0;
        int overstock = 0;
        int safe = 0;

        for (Map<String, String> row : data) {
            String status = row.get("inventory_status");
            if (status == null || status.isBlank()) {
                status = row.get("status");
            }

            if ("REORDER".equalsIgnoreCase(status) || "ORDER NOW".equalsIgnoreCase(status)) {
                reorder++;
            } else if ("OVERSTOCK".equalsIgnoreCase(status)) {
                overstock++;
            } else if ("SAFE".equalsIgnoreCase(status)) {
                safe++;
            }
        }

        Map<String, Integer> result = new HashMap<>();
        result.put("items", data.size());
        result.put("orderNow", reorder);
        result.put("overstock", overstock);
        result.put("safe", safe);

        return result;
    }
}