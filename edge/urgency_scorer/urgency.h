#ifndef URGENCY_H
#define URGENCY_H

typedef struct {
    float safety_flag;
    float obstacle_distance_m;
    float brake_intensity;
    float battery_temp_celsius;
    float sensor_confidence;
    float speed_kmh;
    int ev_state;  /* 0=NORMAL, 1=CAUTION, 2=CRITICAL, 3=EMERGENCY, 4=RECOVERY */
} EVTelemetry;

typedef struct {
    float urgency_score;       /* 0.0 - 1.0 */
    int priority_class;        /* 0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL */
    float component_scores[5]; /* individual component contributions */
} UrgencyResult;

UrgencyResult compute_urgency(const EVTelemetry* telemetry);
float clampf(float val, float min, float max);

#endif /* URGENCY_H */
