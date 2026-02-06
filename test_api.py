#!/usr/bin/env python3
"""
Quick test script for the Video Detector API

Usage:
    python test_api.py [video_path]

If no video path is provided, tests only the health and stats endpoints.
"""

import sys
import requests
import json
from pathlib import Path

API_URL = "http://localhost:5000"

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_health():
    """Test the health endpoint"""
    print_section("Testing Health Endpoint")
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        data = response.json()

        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")

        if response.status_code == 200:
            print("✅ Health check passed!")
            if data.get('model_loaded'):
                print("✅ Model is loaded!")
            else:
                print("⚠️  Model not loaded yet")
            print(f"Device: {data.get('device', 'unknown')}")
            return True
        else:
            print("❌ Health check failed!")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print(f"   Start server with: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_stats():
    """Test the stats endpoint"""
    print_section("Testing Stats Endpoint")
    try:
        response = requests.get(f"{API_URL}/api/stats", timeout=5)
        data = response.json()

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("✅ Stats endpoint working!")
            print(f"\nModel Info:")
            print(f"  Architecture: {data['model_info']['architecture']}")
            print(f"  Accuracy: {data['model_info']['accuracy']}")
            print(f"  F1 Score: {data['model_info']['f1_score']}")
            print(f"  Device: {data['device']}")
            print(f"  Max File Size: {data['max_file_size_mb']} MB")
            print(f"  Supported Formats: {', '.join(data['supported_formats'])}")
            return True
        else:
            print("❌ Stats endpoint failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction(video_path, generate_explanations=False):
    """Test the prediction endpoint"""
    print_section(f"Testing Prediction: {video_path}")

    # Check if file exists
    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        return False

    # Check file size
    file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    if file_size_mb > 100:
        print("⚠️  File is larger than 100MB, may be rejected")

    try:
        # Prepare request
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {'generate_explanations': str(generate_explanations).lower()}

            print(f"Uploading video...")
            print(f"Generate explanations: {generate_explanations}")

            # Send request
            response = requests.post(
                f"{API_URL}/api/predict",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )

        result = response.json()

        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200 and result.get('success'):
            print("✅ Prediction successful!\n")

            pred = result['prediction']
            print(f"Classification: {pred['class']}")
            print(f"Confidence: {pred['confidence']:.2%}")
            print(f"\nProbabilities:")
            print(f"  AI-Generated: {pred['probabilities']['ai_generated']:.2%}")
            print(f"  Real: {pred['probabilities']['real']:.2%}")

            if result.get('attribution_video_url'):
                print(f"\n✅ Attribution video available at:")
                print(f"   {API_URL}{result['attribution_video_url']}")

            print(f"\nVideo ID: {result.get('video_id')}")
            print(f"Timestamp: {result.get('timestamp')}")

            return True
        else:
            print(f"❌ Prediction failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out (took more than 5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_frame_analysis(video_path):
    """Test the frame analysis endpoint"""
    print_section(f"Testing Frame Analysis: {video_path}")

    if not Path(video_path).exists():
        print(f"❌ File not found: {video_path}")
        return False

    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}

            print(f"Analyzing frames...")

            response = requests.post(
                f"{API_URL}/api/analyze/frames",
                files=files,
                timeout=300
            )

        result = response.json()

        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200 and result.get('success'):
            print("✅ Frame analysis successful!\n")

            stats = result['statistics']
            print(f"Frame Statistics:")
            print(f"  Mean importance: {stats['mean']:.3f}")
            print(f"  Std deviation: {stats['std']:.3f}")
            print(f"  Most important frame: {stats['max_frame']}")
            print(f"  Least important frame: {stats['min_frame']}")

            if result.get('key_frames'):
                print(f"\nKey Frames: {', '.join(map(str, result['key_frames']))}")

            return True
        else:
            print(f"❌ Frame analysis failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main test function"""
    print("\n" + "🔍 AI Video Detector API Test Suite".center(60))

    # Parse arguments
    video_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Run tests
    results = []

    # Test health
    results.append(("Health Check", test_health()))

    # Test stats
    results.append(("Stats", test_stats()))

    # Test prediction if video provided
    if video_path:
        results.append(("Prediction", test_prediction(video_path, generate_explanations=False)))
        results.append(("Frame Analysis", test_frame_analysis(video_path)))

        # Optional: test with explanations (slower)
        print("\n" + "⚠️  Skipping attribution test (slow)".center(60))
        print("To test with explanations, run:".center(60))
        print(f"python test_api.py {video_path} --with-explanations".center(60))
    else:
        print("\n" + "ℹ️  No video file provided".center(60))
        print("To test prediction, run:".center(60))
        print("python test_api.py path/to/video.mp4".center(60))

    # Print summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
