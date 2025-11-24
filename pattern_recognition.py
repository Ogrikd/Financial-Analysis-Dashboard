import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import List, Tuple, Dict
import talib

"""
Pattern Recognition and Technical Analysis Module

This module provides:
1. Candlestick Pattern Recognition (Doji, Hammer, Shooting Star, Engulfing)
2. Chart Pattern Recognition (Head and Shoulders)
3. Support/Resistance Level Detection
4. Trend Line Detection
"""


class PatternRecognition:
    """
    Comprehensive pattern recognition for technical analysis
    """

    def __init__(self, df: pd.DataFrame, open_col='Open', high_col='High',
                 low_col='Low', close_col='Close', volume_col='Volume'):
        """
        Initialize with OHLCV data

        Args:
            df: DataFrame with OHLCV data
            open_col, high_col, low_col, close_col, volume_col: Column names
        """
        self.df = df.copy()
        self.open = df[open_col]
        self.high = df[high_col]
        self.low = df[low_col]
        self.close = df[close_col]
        self.volume = df[volume_col] if volume_col in df.columns else None

    # ==================== CANDLESTICK PATTERNS ====================

    def detect_doji(self, threshold=0.1):
        """
        Detect Doji patterns
        Doji: Open and Close are very close (small body)

        Args:
            threshold: Maximum body/range ratio (default 0.1 = 10%)

        Returns:
            Series of boolean values indicating Doji candles
        """
        body = abs(self.close - self.open)
        range_val = self.high - self.low

        # Avoid division by zero
        range_val = range_val.replace(0, np.nan)

        body_ratio = body / range_val
        doji = body_ratio < threshold

        return doji

    def detect_hammer(self, body_ratio=0.3, wick_ratio=2.0):
        """
        Detect Hammer patterns
        Hammer: Small body at top, long lower shadow, little/no upper shadow
        Bullish reversal pattern at bottom of downtrend

        Args:
            body_ratio: Maximum body to range ratio
            wick_ratio: Minimum lower shadow to body ratio

        Returns:
            Series of boolean values
        """
        body = abs(self.close - self.open)
        range_val = self.high - self.low
        lower_shadow = np.minimum(self.open, self.close) - self.low
        upper_shadow = self.high - np.maximum(self.open, self.close)

        # Avoid division by zero
        body = body.replace(0, 0.001)
        range_val = range_val.replace(0, np.nan)

        # Hammer conditions
        small_body = (body / range_val) < body_ratio
        long_lower_shadow = (lower_shadow / body) >= wick_ratio
        small_upper_shadow = upper_shadow < body

        hammer = small_body & long_lower_shadow & small_upper_shadow

        return hammer

    def detect_shooting_star(self, body_ratio=0.3, wick_ratio=2.0):
        """
        Detect Shooting Star patterns
        Shooting Star: Small body at bottom, long upper shadow, little/no lower shadow
        Bearish reversal pattern at top of uptrend

        Args:
            body_ratio: Maximum body to range ratio
            wick_ratio: Minimum upper shadow to body ratio

        Returns:
            Series of boolean values
        """
        body = abs(self.close - self.open)
        range_val = self.high - self.low
        lower_shadow = np.minimum(self.open, self.close) - self.low
        upper_shadow = self.high - np.maximum(self.open, self.close)

        # Avoid division by zero
        body = body.replace(0, 0.001)
        range_val = range_val.replace(0, np.nan)

        # Shooting Star conditions
        small_body = (body / range_val) < body_ratio
        long_upper_shadow = (upper_shadow / body) >= wick_ratio
        small_lower_shadow = lower_shadow < body

        shooting_star = small_body & long_upper_shadow & small_lower_shadow

        return shooting_star

    def detect_bullish_engulfing(self):
        """
        Detect Bullish Engulfing patterns
        Pattern: Previous candle is bearish, current candle is bullish
        and completely engulfs the previous candle's body

        Returns:
            Series of boolean values
        """
        prev_bearish = self.close.shift(1) < self.open.shift(1)
        curr_bullish = self.close > self.open

        engulfs_body = (self.open < self.close.shift(1)) & (self.close > self.open.shift(1))

        bullish_engulfing = prev_bearish & curr_bullish & engulfs_body

        return bullish_engulfing

    def detect_bearish_engulfing(self):
        """
        Detect Bearish Engulfing patterns
        Pattern: Previous candle is bullish, current candle is bearish
        and completely engulfs the previous candle's body

        Returns:
            Series of boolean values
        """
        prev_bullish = self.close.shift(1) > self.open.shift(1)
        curr_bearish = self.close < self.open

        engulfs_body = (self.open > self.close.shift(1)) & (self.close < self.open.shift(1))

        bearish_engulfing = prev_bullish & curr_bearish & engulfs_body

        return bearish_engulfing

    def detect_all_candlestick_patterns(self):
        """
        Detect all candlestick patterns at once

        Returns:
            DataFrame with pattern columns
        """
        patterns = pd.DataFrame(index=self.df.index)

        patterns['Doji'] = self.detect_doji()
        patterns['Hammer'] = self.detect_hammer()
        patterns['Shooting_Star'] = self.detect_shooting_star()
        patterns['Bullish_Engulfing'] = self.detect_bullish_engulfing()
        patterns['Bearish_Engulfing'] = self.detect_bearish_engulfing()

        return patterns

    # ==================== CHART PATTERNS ====================

    def detect_head_and_shoulders(self, window=5, tolerance=0.03):
        """
        Detect Head and Shoulders pattern
        Pattern: Three peaks - left shoulder, head (highest), right shoulder

        Args:
            window: Window for finding local maxima
            tolerance: Price tolerance for shoulder symmetry (3% default)

        Returns:
            List of dictionaries with pattern information
        """
        # Find local maxima (peaks)
        peaks_idx = argrelextrema(self.high.values, np.greater, order=window)[0]

        # Find local minima (troughs)
        troughs_idx = argrelextrema(self.low.values, np.less, order=window)[0]

        patterns = []

        # Need at least 3 peaks for H&S
        if len(peaks_idx) < 3:
            return patterns

        # Check each set of 3 consecutive peaks
        for i in range(len(peaks_idx) - 2):
            left_shoulder_idx = peaks_idx[i]
            head_idx = peaks_idx[i + 1]
            right_shoulder_idx = peaks_idx[i + 2]

            left_shoulder = self.high.iloc[left_shoulder_idx]
            head = self.high.iloc[head_idx]
            right_shoulder = self.high.iloc[right_shoulder_idx]

            # Head should be higher than both shoulders
            if head > left_shoulder and head > right_shoulder:
                # Shoulders should be roughly equal (within tolerance)
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder

                if shoulder_diff < tolerance:
                    # Find neckline (trough between left shoulder and head, and head and right shoulder)
                    relevant_troughs = troughs_idx[
                        (troughs_idx > left_shoulder_idx) &
                        (troughs_idx < right_shoulder_idx)
                        ]

                    if len(relevant_troughs) >= 2:
                        neckline_left_idx = relevant_troughs[0]
                        neckline_right_idx = relevant_troughs[-1]

                        neckline_left = self.low.iloc[neckline_left_idx]
                        neckline_right = self.low.iloc[neckline_right_idx]

                        patterns.append({
                            'type': 'Head_and_Shoulders',
                            'left_shoulder': (left_shoulder_idx, left_shoulder),
                            'head': (head_idx, head),
                            'right_shoulder': (right_shoulder_idx, right_shoulder),
                            'neckline_left': (neckline_left_idx, neckline_left),
                            'neckline_right': (neckline_right_idx, neckline_right),
                            'completed': right_shoulder_idx < len(self.df) - 1
                        })

        return patterns

    def detect_inverse_head_and_shoulders(self, window=5, tolerance=0.03):
        """
        Detect Inverse Head and Shoulders pattern (bullish)
        Pattern: Three troughs - left shoulder, head (lowest), right shoulder

        Args:
            window: Window for finding local minima
            tolerance: Price tolerance for shoulder symmetry

        Returns:
            List of dictionaries with pattern information
        """
        # Find local minima (troughs)
        troughs_idx = argrelextrema(self.low.values, np.less, order=window)[0]

        # Find local maxima (peaks)
        peaks_idx = argrelextrema(self.high.values, np.greater, order=window)[0]

        patterns = []

        if len(troughs_idx) < 3:
            return patterns

        for i in range(len(troughs_idx) - 2):
            left_shoulder_idx = troughs_idx[i]
            head_idx = troughs_idx[i + 1]
            right_shoulder_idx = troughs_idx[i + 2]

            left_shoulder = self.low.iloc[left_shoulder_idx]
            head = self.low.iloc[head_idx]
            right_shoulder = self.low.iloc[right_shoulder_idx]

            # Head should be lower than both shoulders
            if head < left_shoulder and head < right_shoulder:
                shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder

                if shoulder_diff < tolerance:
                    relevant_peaks = peaks_idx[
                        (peaks_idx > left_shoulder_idx) &
                        (peaks_idx < right_shoulder_idx)
                        ]

                    if len(relevant_peaks) >= 2:
                        neckline_left_idx = relevant_peaks[0]
                        neckline_right_idx = relevant_peaks[-1]

                        neckline_left = self.high.iloc[neckline_left_idx]
                        neckline_right = self.high.iloc[neckline_right_idx]

                        patterns.append({
                            'type': 'Inverse_Head_and_Shoulders',
                            'left_shoulder': (left_shoulder_idx, left_shoulder),
                            'head': (head_idx, head),
                            'right_shoulder': (right_shoulder_idx, right_shoulder),
                            'neckline_left': (neckline_left_idx, neckline_left),
                            'neckline_right': (neckline_right_idx, neckline_right),
                            'completed': right_shoulder_idx < len(self.df) - 1
                        })

        return patterns

    # ==================== SUPPORT/RESISTANCE ====================

    def detect_support_resistance(self, window=20, num_levels=3):
        """
        Detect support and resistance levels using local extrema

        Args:
            window: Window size for finding local extrema
            num_levels: Number of top support/resistance levels to return

        Returns:
            Dictionary with support and resistance levels
        """
        # Find local maxima (resistance)
        resistance_idx = argrelextrema(self.high.values, np.greater, order=window)[0]
        resistance_levels = self.high.iloc[resistance_idx].values

        # Find local minima (support)
        support_idx = argrelextrema(self.low.values, np.less, order=window)[0]
        support_levels = self.low.iloc[support_idx].values

        # Cluster nearby levels
        resistance_clusters = self._cluster_levels(resistance_levels, num_levels)
        support_clusters = self._cluster_levels(support_levels, num_levels)

        return {
            'resistance': resistance_clusters,
            'support': support_clusters,
            'resistance_indices': resistance_idx.tolist(),
            'support_indices': support_idx.tolist()
        }

    def _cluster_levels(self, levels, num_clusters):
        """
        Cluster price levels that are close together

        Args:
            levels: Array of price levels
            num_clusters: Number of clusters to return

        Returns:
            Array of clustered price levels
        """
        if len(levels) == 0:
            return []

        # Sort levels
        sorted_levels = np.sort(levels)

        # Simple clustering by proximity
        clusters = []
        current_cluster = [sorted_levels[0]]

        tolerance = np.std(sorted_levels) * 0.5  # 0.5 standard deviations

        for level in sorted_levels[1:]:
            if abs(level - np.mean(current_cluster)) < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]

        # Add last cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))

        # Return top N clusters by frequency/strength
        # For now, return evenly spaced clusters
        if len(clusters) > num_clusters:
            step = len(clusters) // num_clusters
            return [clusters[i * step] for i in range(num_clusters)]

        return clusters

    # ==================== TREND LINES ====================

    def detect_trend_lines(self, window=14, min_touches=2):
        """
        Detect uptrend and downtrend lines programmatically

        Args:
            window: Lookback window for trend detection
            min_touches: Minimum number of price touches for valid trend line

        Returns:
            Dictionary with uptrend and downtrend line parameters
        """
        # Find pivot points
        highs_idx = argrelextrema(self.high.values, np.greater, order=5)[0]
        lows_idx = argrelextrema(self.low.values, np.less, order=5)[0]

        uptrend_lines = []
        downtrend_lines = []

        # Detect uptrend lines (connecting higher lows)
        if len(lows_idx) >= 2:
            for i in range(len(lows_idx) - 1):
                for j in range(i + 1, len(lows_idx)):
                    idx1, idx2 = lows_idx[i], lows_idx[j]
                    price1, price2 = self.low.iloc[idx1], self.low.iloc[idx2]

                    # Check if line is ascending
                    if price2 > price1:
                        # Calculate line parameters
                        slope = (price2 - price1) / (idx2 - idx1)
                        intercept = price1 - slope * idx1

                        # Count touches
                        touches = self._count_line_touches(
                            self.low, slope, intercept, idx1, idx2, tolerance=0.02
                        )

                        if touches >= min_touches:
                            uptrend_lines.append({
                                'start_idx': idx1,
                                'end_idx': idx2,
                                'start_price': price1,
                                'end_price': price2,
                                'slope': slope,
                                'intercept': intercept,
                                'touches': touches
                            })

        # Detect downtrend lines (connecting lower highs)
        if len(highs_idx) >= 2:
            for i in range(len(highs_idx) - 1):
                for j in range(i + 1, len(highs_idx)):
                    idx1, idx2 = highs_idx[i], highs_idx[j]
                    price1, price2 = self.high.iloc[idx1], self.high.iloc[idx2]

                    # Check if line is descending
                    if price2 < price1:
                        slope = (price2 - price1) / (idx2 - idx1)
                        intercept = price1 - slope * idx1

                        touches = self._count_line_touches(
                            self.high, slope, intercept, idx1, idx2, tolerance=0.02
                        )

                        if touches >= min_touches:
                            downtrend_lines.append({
                                'start_idx': idx1,
                                'end_idx': idx2,
                                'start_price': price1,
                                'end_price': price2,
                                'slope': slope,
                                'intercept': intercept,
                                'touches': touches
                            })

        # Sort by number of touches and recency
        uptrend_lines = sorted(uptrend_lines, key=lambda x: (x['touches'], x['end_idx']), reverse=True)
        downtrend_lines = sorted(downtrend_lines, key=lambda x: (x['touches'], x['end_idx']), reverse=True)

        return {
            'uptrend': uptrend_lines[:3],  # Top 3 uptrend lines
            'downtrend': downtrend_lines[:3]  # Top 3 downtrend lines
        }

    def _count_line_touches(self, prices, slope, intercept, start_idx, end_idx, tolerance=0.02):
        """
        Count how many times price touches a trend line

        Args:
            prices: Price series (high or low)
            slope: Line slope
            intercept: Line intercept
            start_idx, end_idx: Line start and end indices
            tolerance: Price tolerance (2% default)

        Returns:
            Number of touches
        """
        touches = 0

        for idx in range(start_idx, min(end_idx + 1, len(prices))):
            expected_price = slope * idx + intercept
            actual_price = prices.iloc[idx]

            # Check if price is within tolerance
            if abs(actual_price - expected_price) / expected_price < tolerance:
                touches += 1

        return touches

    # ==================== COMPREHENSIVE ANALYSIS ====================

    def analyze_all_patterns(self):
        """
        Run all pattern detection methods and return comprehensive results

        Returns:
            Dictionary with all detected patterns
        """
        results = {
            'candlestick_patterns': self.detect_all_candlestick_patterns(),
            'head_and_shoulders': self.detect_head_and_shoulders(),
            'inverse_head_and_shoulders': self.detect_inverse_head_and_shoulders(),
            'support_resistance': self.detect_support_resistance(),
            'trend_lines': self.detect_trend_lines()
        }

        return results


# ==================== UTILITY FUNCTIONS ====================

def get_pattern_summary(patterns_df: pd.DataFrame, recent_bars=20):
    """
    Get summary of recent patterns

    Args:
        patterns_df: DataFrame from detect_all_candlestick_patterns()
        recent_bars: Number of recent bars to check

    Returns:
        Dictionary with pattern counts and latest occurrences
    """
    recent = patterns_df.tail(recent_bars)

    summary = {}
    for col in patterns_df.columns:
        count = recent[col].sum()
        if count > 0:
            latest_idx = recent[recent[col]].index[-1]
            summary[col] = {
                'count': int(count),
                'latest': str(latest_idx)
            }

    return summary


def format_pattern_results(pattern_recognition: PatternRecognition, num_recent=50):
    """
    Format pattern recognition results for display

    Args:
        pattern_recognition: PatternRecognition instance
        num_recent: Number of recent bars to analyze

    Returns:
        Formatted dictionary of results
    """
    results = pattern_recognition.analyze_all_patterns()

    # Get recent candlestick patterns
    candlestick_patterns = results['candlestick_patterns'].tail(num_recent)
    pattern_summary = get_pattern_summary(candlestick_patterns, num_recent)

    formatted = {
        'candlestick_summary': pattern_summary,
        'candlestick_latest': {},
        'chart_patterns': {
            'head_and_shoulders': len(results['head_and_shoulders']),
            'inverse_head_and_shoulders': len(results['inverse_head_and_shoulders'])
        },
        'support_levels': results['support_resistance']['support'],
        'resistance_levels': results['support_resistance']['resistance'],
        'trend_lines': {
            'uptrend_count': len(results['trend_lines']['uptrend']),
            'downtrend_count': len(results['trend_lines']['downtrend'])
        }
    }

    # Get latest occurrence of each pattern
    for col in candlestick_patterns.columns:
        if candlestick_patterns[col].any():
            latest_idx = candlestick_patterns[candlestick_patterns[col]].index[-1]
            formatted['candlestick_latest'][col] = str(latest_idx)

    return formatted