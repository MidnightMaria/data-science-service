package com.agnesmaria.datascience.dto;

public class DashboardSummaryResponse {

    private int totalSeries;
    private int improvedSeries;
    private double improvementPct;

    private int reorder;
    private int safe;
    private int overstock;

    private double avgSafetyStock;
    private double avgReorderPoint;
    private double avgEoq;

    public DashboardSummaryResponse() {
    }

    public int getTotalSeries() {
        return totalSeries;
    }

    public void setTotalSeries(int totalSeries) {
        this.totalSeries = totalSeries;
    }

    public int getImprovedSeries() {
        return improvedSeries;
    }

    public void setImprovedSeries(int improvedSeries) {
        this.improvedSeries = improvedSeries;
    }

    public double getImprovementPct() {
        return improvementPct;
    }

    public void setImprovementPct(double improvementPct) {
        this.improvementPct = improvementPct;
    }

    public int getReorder() {
        return reorder;
    }

    public void setReorder(int reorder) {
        this.reorder = reorder;
    }

    public int getSafe() {
        return safe;
    }

    public void setSafe(int safe) {
        this.safe = safe;
    }

    public int getOverstock() {
        return overstock;
    }

    public void setOverstock(int overstock) {
        this.overstock = overstock;
    }

    public double getAvgSafetyStock() {
        return avgSafetyStock;
    }

    public void setAvgSafetyStock(double avgSafetyStock) {
        this.avgSafetyStock = avgSafetyStock;
    }

    public double getAvgReorderPoint() {
        return avgReorderPoint;
    }

    public void setAvgReorderPoint(double avgReorderPoint) {
        this.avgReorderPoint = avgReorderPoint;
    }

    public double getAvgEoq() {
        return avgEoq;
    }

    public void setAvgEoq(double avgEoq) {
        this.avgEoq = avgEoq;
    }
}