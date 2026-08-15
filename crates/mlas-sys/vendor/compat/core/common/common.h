// FEASIBILITY SPIKE shim: q4common.h includes "core/common/common.h" but uses
// nothing from it in the SGEMM/Q4-dispatch path we compile. Empty stand-in to
// avoid pulling ORT core/common (logging, status, exceptions, GSL, ...).
#pragma once
