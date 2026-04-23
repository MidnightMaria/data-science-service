package com.agnesmaria.datascience.dto;

import java.util.List;

public class ModelEvaluationResponse {

    private Summary summary;
    private List<ModelMetric> comparisonTable;
    private List<SeriesMetric> bestSeries;
    private List<SeriesMetric> worstSeries;

    public ModelEvaluationResponse() {
    }

    public Summary getSummary() {
        return summary;
    }

    public void setSummary(Summary summary) {
        this.summary = summary;
    }

    public List<ModelMetric> getComparisonTable() {
        return comparisonTable;
    }

    public void setComparisonTable(List<ModelMetric> comparisonTable) {
        this.comparisonTable = comparisonTable;
    }

    public List<SeriesMetric> getBestSeries() {
        return bestSeries;
    }

    public void setBestSeries(List<SeriesMetric> bestSeries) {
        this.bestSeries = bestSeries;
    }

    public List<SeriesMetric> getWorstSeries() {
        return worstSeries;
    }

    public void setWorstSeries(List<SeriesMetric> worstSeries) {
        this.worstSeries = worstSeries;
    }

    public static class Summary {
        private double hybridMae;
        private double hybridRmse;
        private double hybridSmape;
        private double prophetMae;
        private double prophetRmse;
        private double prophetSmape;
        private int improvedSeries;
        private int totalSeries;
        private double improvementPct;

        public Summary() {
        }

        public double getHybridMae() {
            return hybridMae;
        }

        public void setHybridMae(double hybridMae) {
            this.hybridMae = hybridMae;
        }

        public double getHybridRmse() {
            return hybridRmse;
        }

        public void setHybridRmse(double hybridRmse) {
            this.hybridRmse = hybridRmse;
        }

        public double getHybridSmape() {
            return hybridSmape;
        }

        public void setHybridSmape(double hybridSmape) {
            this.hybridSmape = hybridSmape;
        }

        public double getProphetMae() {
            return prophetMae;
        }

        public void setProphetMae(double prophetMae) {
            this.prophetMae = prophetMae;
        }

        public double getProphetRmse() {
            return prophetRmse;
        }

        public void setProphetRmse(double prophetRmse) {
            this.prophetRmse = prophetRmse;
        }

        public double getProphetSmape() {
            return prophetSmape;
        }

        public void setProphetSmape(double prophetSmape) {
            this.prophetSmape = prophetSmape;
        }

        public int getImprovedSeries() {
            return improvedSeries;
        }

        public void setImprovedSeries(int improvedSeries) {
            this.improvedSeries = improvedSeries;
        }

        public int getTotalSeries() {
            return totalSeries;
        }

        public void setTotalSeries(int totalSeries) {
            this.totalSeries = totalSeries;
        }

        public double getImprovementPct() {
            return improvementPct;
        }

        public void setImprovementPct(double improvementPct) {
            this.improvementPct = improvementPct;
        }
    }

    public static class ModelMetric {
        private String model;
        private double mae;
        private double rmse;
        private double mape;
        private double smape;

        public ModelMetric() {
        }

        public String getModel() {
            return model;
        }

        public void setModel(String model) {
            this.model = model;
        }

        public double getMae() {
            return mae;
        }

        public void setMae(double mae) {
            this.mae = mae;
        }

        public double getRmse() {
            return rmse;
        }

        public void setRmse(double rmse) {
            this.rmse = rmse;
        }

        public double getMape() {
            return mape;
        }

        public void setMape(double mape) {
            this.mape = mape;
        }

        public double getSmape() {
            return smape;
        }

        public void setSmape(double smape) {
            this.smape = smape;
        }
    }

    public static class SeriesMetric {
        private int store;
        private int item;
        private double prophet;
        private double hybrid;
        private double improvement;

        public SeriesMetric() {
        }

        public int getStore() {
            return store;
        }

        public void setStore(int store) {
            this.store = store;
        }

        public int getItem() {
            return item;
        }

        public void setItem(int item) {
            this.item = item;
        }

        public double getProphet() {
            return prophet;
        }

        public void setProphet(double prophet) {
            this.prophet = prophet;
        }

        public double getHybrid() {
            return hybrid;
        }

        public void setHybrid(double hybrid) {
            this.hybrid = hybrid;
        }

        public double getImprovement() {
            return improvement;
        }

        public void setImprovement(double improvement) {
            this.improvement = improvement;
        }
    }
}