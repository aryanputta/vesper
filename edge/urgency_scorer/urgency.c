/*
 * VESPER – Edge Urgency Scorer
 * urgency.c – Composite urgency scoring for EV telemetry.
 *
 * Compile as shared library:
 *   gcc -O2 -Wall -fPIC -shared -o urgency_scorer.so urgency.c -lm
 *
 * Compile for standalone test:
 *   gcc -O2 -Wall -DTEST_MAIN -o urgency_test urgency.c -lm
 */

#include <stdio.h>
#include "urgency.h"

/* ---------------------------------------------------------------------------
 * clampf
 * --------------------------------------------------------------------------- */
float clampf(float val, float min, float max)
{
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

/* ---------------------------------------------------------------------------
 * compute_urgency
 *
 * Weights:
 *   W_SAFETY   = 0.30
 *   W_OBSTACLE = 0.25
 *   W_BRAKE    = 0.20
 *   W_BATTERY  = 0.15
 *   W_SENSOR   = 0.10
 *
 * State multipliers:
 *   NORMAL(0)    = 1.0
 *   CAUTION(1)   = 1.1
 *   CRITICAL(2)  = 1.3
 *   EMERGENCY(3) = 1.5
 *   RECOVERY(4)  = 1.0
 * --------------------------------------------------------------------------- */
UrgencyResult compute_urgency(const EVTelemetry* telemetry)
{
    UrgencyResult result;

    /* ── Weights ──────────────────────────────────────────────────────────── */
    const float W_SAFETY   = 0.30f;
    const float W_OBSTACLE = 0.25f;
    const float W_BRAKE    = 0.20f;
    const float W_BATTERY  = 0.15f;
    const float W_SENSOR   = 0.10f;

    /* ── Component scores ─────────────────────────────────────────────────── */

    /* Safety flag: already 0 or 1 from the caller */
    float safety_score = clampf(telemetry->safety_flag, 0.0f, 1.0f);

    /* Obstacle: score is higher the closer the obstacle is (cap at 50 m) */
    float obstacle_score;
    if (telemetry->obstacle_distance_m < 50.0f) {
        obstacle_score = clampf(
            (50.0f - telemetry->obstacle_distance_m) / 50.0f,
            0.0f, 1.0f
        );
    } else {
        obstacle_score = 0.0f;
    }

    /* Brake intensity: already normalised [0, 1] */
    float brake_score = clampf(telemetry->brake_intensity, 0.0f, 1.0f);

    /* Battery temperature: linear ramp from 40 °C (score=0) to 80 °C (score=1) */
    float battery_score = clampf(
        (telemetry->battery_temp_celsius - 40.0f) / 40.0f,
        0.0f, 1.0f
    );

    /* Sensor confidence: invert so low confidence → high urgency */
    float sensor_score = clampf(1.0f - telemetry->sensor_confidence, 0.0f, 1.0f);

    /* ── Weighted sum ─────────────────────────────────────────────────────── */
    float weighted_sum = (W_SAFETY   * safety_score)
                       + (W_OBSTACLE * obstacle_score)
                       + (W_BRAKE    * brake_score)
                       + (W_BATTERY  * battery_score)
                       + (W_SENSOR   * sensor_score);

    /* ── State multiplier ─────────────────────────────────────────────────── */
    float state_multiplier;
    switch (telemetry->ev_state) {
        case 0:  state_multiplier = 1.0f; break; /* NORMAL    */
        case 1:  state_multiplier = 1.1f; break; /* CAUTION   */
        case 2:  state_multiplier = 1.3f; break; /* CRITICAL  */
        case 3:  state_multiplier = 1.5f; break; /* EMERGENCY */
        case 4:  state_multiplier = 1.0f; break; /* RECOVERY  */
        default: state_multiplier = 1.0f; break;
    }

    /* ── Final score ──────────────────────────────────────────────────────── */
    float final_score = clampf(weighted_sum * state_multiplier, 0.0f, 1.0f);

    /* ── Priority class ───────────────────────────────────────────────────── */
    int priority_class;
    if      (final_score >= 0.75f) priority_class = 3; /* CRITICAL */
    else if (final_score >= 0.50f) priority_class = 2; /* HIGH     */
    else if (final_score >= 0.25f) priority_class = 1; /* MEDIUM   */
    else                           priority_class = 0; /* LOW      */

    /* ── Populate result ──────────────────────────────────────────────────── */
    result.urgency_score        = final_score;
    result.priority_class       = priority_class;
    result.component_scores[0]  = safety_score;
    result.component_scores[1]  = obstacle_score;
    result.component_scores[2]  = brake_score;
    result.component_scores[3]  = battery_score;
    result.component_scores[4]  = sensor_score;

    return result;
}

/* ---------------------------------------------------------------------------
 * Optional self-test main (compiled with -DTEST_MAIN)
 * --------------------------------------------------------------------------- */
#ifdef TEST_MAIN

#include <assert.h>
#include <string.h>

static void print_result(const char* label, const UrgencyResult* r)
{
    printf("[%s] score=%.4f priority=%d  "
           "components=[safety=%.3f obstacle=%.3f brake=%.3f battery=%.3f sensor=%.3f]\n",
           label,
           r->urgency_score,
           r->priority_class,
           r->component_scores[0],
           r->component_scores[1],
           r->component_scores[2],
           r->component_scores[3],
           r->component_scores[4]);
}

int main(void)
{
    EVTelemetry t;
    UrgencyResult r;

    /* ── Test 1: All-zero telemetry ─────────────────────────────────────── */
    memset(&t, 0, sizeof(t));
    t.obstacle_distance_m  = 999.0f;
    t.sensor_confidence    = 1.0f;
    t.battery_temp_celsius = 20.0f;
    t.ev_state             = 0; /* NORMAL */
    r = compute_urgency(&t);
    print_result("all-zero", &r);
    assert(r.urgency_score == 0.0f && "all-zero should give score 0");
    assert(r.priority_class == 0   && "all-zero should give LOW priority");

    /* ── Test 2: Emergency scenario ─────────────────────────────────────── */
    t.safety_flag          = 1.0f;
    t.obstacle_distance_m  = 3.0f;
    t.brake_intensity      = 0.95f;
    t.battery_temp_celsius = 75.0f;
    t.sensor_confidence    = 0.1f;
    t.ev_state             = 3; /* EMERGENCY */
    r = compute_urgency(&t);
    print_result("emergency", &r);
    assert(r.urgency_score == 1.0f    && "emergency should clamp to 1.0");
    assert(r.priority_class == 3      && "emergency should give CRITICAL priority");

    /* ── Test 3: Normal driving ──────────────────────────────────────────── */
    t.safety_flag          = 0.0f;
    t.obstacle_distance_m  = 80.0f;
    t.brake_intensity      = 0.05f;
    t.battery_temp_celsius = 35.0f;
    t.sensor_confidence    = 0.95f;
    t.ev_state             = 0; /* NORMAL */
    r = compute_urgency(&t);
    print_result("normal-drive", &r);
    assert(r.priority_class == 0 && "normal driving should be LOW priority");

    /* ── Test 4: Battery overheat in CRITICAL state ──────────────────────── */
    t.safety_flag          = 0.0f;
    t.obstacle_distance_m  = 999.0f;
    t.brake_intensity      = 0.0f;
    t.battery_temp_celsius = 68.0f; /* above 40+40=80 → battery_score = 0.7 */
    t.sensor_confidence    = 1.0f;
    t.ev_state             = 2; /* CRITICAL – multiplier 1.3 */
    r = compute_urgency(&t);
    print_result("battery-overheat-critical", &r);

    /* Expected: battery_score = (68-40)/40 = 0.7
     *   weighted = 0.15 * 0.7 = 0.105
     *   final    = 0.105 * 1.3 = 0.1365 → MEDIUM? still LOW<0.25 → LOW
     *   Actually 0.1365 < 0.25 → LOW (0)
     */
    assert(r.priority_class == 0 && "battery only in critical should still be LOW with no other triggers");

    /* ── Test 5: clampf edge cases ───────────────────────────────────────── */
    assert(clampf(-5.0f, 0.0f, 1.0f) == 0.0f);
    assert(clampf(2.5f,  0.0f, 1.0f) == 1.0f);
    assert(clampf(0.5f,  0.0f, 1.0f) == 0.5f);

    printf("\nAll tests passed.\n");
    return 0;
}

#endif /* TEST_MAIN */
