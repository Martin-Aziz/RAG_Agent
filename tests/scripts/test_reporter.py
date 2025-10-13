#!/usr/bin/env python3
"""
Test Reporter: Generates comprehensive test summaries and reports.

This script:
- Parses JUnit XML files from test runs
- Extracts coverage data from coverage.xml
- Generates summary statistics
- Creates HTML reports with charts
- Outputs JSON summary for CI/CD
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import xml.etree.ElementTree as ET
from datetime import datetime


class TestReporter:
    """Generates test reports from pytest output."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "all_passed": True,
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "duration": 0.0,
            "coverage": 0.0,
            "categories": {}
        }
    
    def parse_junit_xml(self, xml_path: Path, category: str) -> Dict[str, Any]:
        """Parse JUnit XML file and extract test results."""
        if not xml_path.exists():
            print(f"⚠️  Warning: {xml_path} not found, skipping")
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Handle both <testsuites> and <testsuite> root elements
            if root.tag == "testsuites":
                testsuites = root.findall("testsuite")
            else:
                testsuites = [root]
            
            stats = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0.0,
                "failures": []
            }
            
            for suite in testsuites:
                stats["total"] += int(suite.get("tests", 0))
                stats["failed"] += int(suite.get("failures", 0))
                stats["skipped"] += int(suite.get("skipped", 0))
                stats["errors"] += int(suite.get("errors", 0))
                stats["duration"] += float(suite.get("time", 0))
                
                # Collect failure details
                for testcase in suite.findall("testcase"):
                    failure = testcase.find("failure")
                    error = testcase.find("error")
                    
                    if failure is not None:
                        stats["failures"].append({
                            "name": testcase.get("name"),
                            "classname": testcase.get("classname"),
                            "message": failure.get("message", ""),
                            "type": "failure"
                        })
                    elif error is not None:
                        stats["failures"].append({
                            "name": testcase.get("name"),
                            "classname": testcase.get("classname"),
                            "message": error.get("message", ""),
                            "type": "error"
                        })
            
            stats["passed"] = stats["total"] - stats["failed"] - stats["skipped"] - stats["errors"]
            
            print(f"✅ Parsed {category}: {stats['total']} tests, "
                  f"{stats['passed']} passed, {stats['failed']} failed, "
                  f"{stats['skipped']} skipped")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error parsing {xml_path}: {e}")
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    
    def parse_coverage_xml(self, xml_path: Path) -> float:
        """Parse coverage.xml and extract line coverage percentage."""
        if not xml_path.exists():
            print(f"⚠️  Warning: {xml_path} not found, coverage unavailable")
            return 0.0
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Coverage XML has format: <coverage line-rate="0.87" ...>
            line_rate = float(root.get("line-rate", 0))
            coverage_pct = line_rate * 100
            
            print(f"📊 Coverage: {coverage_pct:.2f}%")
            return coverage_pct
            
        except Exception as e:
            print(f"❌ Error parsing {xml_path}: {e}")
            return 0.0
    
    def collect_all_results(self) -> None:
        """Collect results from all test categories."""
        categories = {
            "unit": "junit-unit.xml",
            "integration": "junit-integration.xml",
            "blackbox": "junit-blackbox.xml",
            "behavior": "junit-behavior.xml",
            "adversarial": "junit-adversarial.xml",
            "performance": "junit-performance.xml",
            "regression": "junit-regression.xml"
        }
        
        for category, filename in categories.items():
            xml_path = self.reports_dir / filename
            stats = self.parse_junit_xml(xml_path, category)
            
            self.summary["categories"][category] = stats
            self.summary["total_tests"] += stats["total"]
            self.summary["total_passed"] += stats["passed"]
            self.summary["total_failed"] += stats["failed"]
            self.summary["total_skipped"] += stats["skipped"]
            self.summary["duration"] += stats.get("duration", 0)
            
            if stats["failed"] > 0 or stats["errors"] > 0:
                self.summary["all_passed"] = False
        
        # Parse coverage
        coverage_path = self.reports_dir / "coverage.xml"
        self.summary["coverage"] = self.parse_coverage_xml(coverage_path)
    
    def generate_json_summary(self) -> None:
        """Save summary as JSON for CI/CD consumption."""
        output_path = self.reports_dir / "summary.json"
        
        with open(output_path, "w") as f:
            json.dump(self.summary, f, indent=2)
        
        print(f"\n📄 JSON summary saved to: {output_path}")
    
    def generate_console_report(self) -> None:
        """Print a formatted console report."""
        print("\n" + "="*80)
        print("🧪 RAG AGENT TEST SUITE SUMMARY")
        print("="*80)
        print(f"Timestamp: {self.summary['timestamp']}")
        print(f"Status: {'✅ ALL PASSED' if self.summary['all_passed'] else '❌ FAILED'}")
        print(f"\nOverall:")
        print(f"  Total Tests:   {self.summary['total_tests']}")
        print(f"  Passed:        {self.summary['total_passed']} ({self._percentage(self.summary['total_passed'], self.summary['total_tests'])}%)")
        print(f"  Failed:        {self.summary['total_failed']}")
        print(f"  Skipped:       {self.summary['total_skipped']}")
        print(f"  Duration:      {self.summary['duration']:.2f}s")
        print(f"  Coverage:      {self.summary['coverage']:.2f}%")
        
        print("\nBy Category:")
        print(f"{'Category':<15} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Skipped':<8} {'Status':<10}")
        print("-" * 80)
        
        for category, stats in self.summary["categories"].items():
            status = "✅ PASS" if stats["failed"] == 0 and stats["errors"] == 0 else "❌ FAIL"
            print(f"{category:<15} {stats['total']:<8} {stats['passed']:<8} "
                  f"{stats['failed']:<8} {stats['skipped']:<8} {status:<10}")
        
        # Show failures
        if not self.summary["all_passed"]:
            print("\n❌ Failed Tests:")
            print("-" * 80)
            for category, stats in self.summary["categories"].items():
                if stats.get("failures"):
                    print(f"\n{category.upper()}:")
                    for failure in stats["failures"]:
                        print(f"  • {failure['classname']}.{failure['name']}")
                        print(f"    {failure['message'][:100]}...")
        
        print("\n" + "="*80)
    
    def generate_html_report(self) -> None:
        """Generate an HTML report with charts."""
        html_path = self.reports_dir / "summary.html"
        
        # Prepare data for charts
        categories = list(self.summary["categories"].keys())
        passed = [self.summary["categories"][cat]["passed"] for cat in categories]
        failed = [self.summary["categories"][cat]["failed"] for cat in categories]
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Agent Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: {'#4caf50' if self.summary['all_passed'] else '#f44336'};
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .status {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        .pass {{ color: #4caf50; }}
        .fail {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="status">{'✅ ALL TESTS PASSED' if self.summary['all_passed'] else '❌ TESTS FAILED'}</div>
        <div>{self.summary['timestamp']}</div>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{self.summary['total_tests']}</div>
            <div class="stat-label">Total Tests</div>
        </div>
        <div class="stat-card">
            <div class="stat-value pass">{self.summary['total_passed']}</div>
            <div class="stat-label">Passed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value fail">{self.summary['total_failed']}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{self.summary['coverage']:.1f}%</div>
            <div class="stat-label">Coverage</div>
        </div>
    </div>
    
    <div class="chart-container">
        <canvas id="categoryChart"></canvas>
    </div>
    
    <div class="chart-container">
        <h2>Test Results by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Total</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Skipped</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for category, stats in self.summary["categories"].items():
            status_class = "pass" if stats["failed"] == 0 else "fail"
            status_text = "✅ PASS" if stats["failed"] == 0 else "❌ FAIL"
            html_content += f"""
                <tr>
                    <td><strong>{category.title()}</strong></td>
                    <td>{stats['total']}</td>
                    <td class="pass">{stats['passed']}</td>
                    <td class="fail">{stats['failed']}</td>
                    <td>{stats['skipped']}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
"""
        
        html_content += f"""
            </tbody>
        </table>
    </div>
    
    <script>
        const ctx = document.getElementById('categoryChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(categories)},
                datasets: [
                    {{
                        label: 'Passed',
                        data: {json.dumps(passed)},
                        backgroundColor: 'rgba(76, 175, 80, 0.8)'
                    }},
                    {{
                        label: 'Failed',
                        data: {json.dumps(failed)},
                        backgroundColor: 'rgba(244, 67, 54, 0.8)'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ stacked: true }},
                    y: {{ stacked: true, beginAtZero: true }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Test Results by Category',
                        font: {{ size: 18 }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        with open(html_path, "w") as f:
            f.write(html_content)
        
        print(f"📊 HTML report saved to: {html_path}")
    
    def _percentage(self, value: int, total: int) -> float:
        """Calculate percentage safely."""
        return (value / total * 100) if total > 0 else 0.0
    
    def run(self) -> int:
        """Run the full report generation and return exit code."""
        print("🔍 Collecting test results...")
        self.collect_all_results()
        
        print("\n📝 Generating reports...")
        self.generate_json_summary()
        self.generate_html_report()
        self.generate_console_report()
        
        # Return non-zero exit code if tests failed
        return 0 if self.summary["all_passed"] else 1


if __name__ == "__main__":
    # Allow custom reports directory
    reports_dir = sys.argv[1] if len(sys.argv) > 1 else "reports"
    
    reporter = TestReporter(reports_dir)
    exit_code = reporter.run()
    
    sys.exit(exit_code)
