"""api/schemas.py — Modelos Pydantic para validación de entrada y salida."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    """Datos de entrada del clasificador de renovación de préstamo.

    Los nombres de campo replican las columnas crudas del dataset original
    (antes del renombrado interno), para que el payload sea el mismo que
    produce el sistema origen de datos del banco.
    """
    MES: int = Field(..., description="Periodo (formato AAAAMM)")
    CLIENTE: int = Field(..., description="Identificador del cliente")
    LINEA_RENOVADO: float = Field(..., description="Línea de crédito ofrecida en la renovación")
    PLAZO_RENOVADO: int = Field(..., description="Plazo ofrecido en meses")
    USO_LINEA_TOTAL_TC_T2: float = Field(..., description="Uso de línea de TC (T2), 0-1")
    USO_TRIM_LINEA_BBVA: float = Field(..., description="Uso trimestral de línea, 0-1")
    NR_ENTIDADES_TOTAL_T2: int = Field(..., description="Número de entidades financieras (T2)")
    DIFF_NRO_ENTIDA_TOTALES_T2_T12: int = Field(..., description="Diferencia de entidades T2 vs T12")
    SDO_CONSUMO_T2: float = Field(..., description="Saldo de consumo (T2)")
    RESENCIA_OFERTA_PLD_RENOVADO: Optional[float] = Field(None, description="Meses desde la última oferta")
    Ahorro_Sldo_Bco_T1: float = Field(..., description="Saldo de ahorros (T1)")
    PConsumo_Sldo_Bco_T1: float = Field(..., description="Préstamo de consumo vigente (T1)")
    SDO_BCO_tot_sm_pasivo_Bco_6M: float = Field(..., description="Promedio de deuda bancaria últimos 6 meses")
    EDAD: Optional[float] = Field(None, description="Edad del cliente")
    SEXO: Optional[str] = Field(None, description="Sexo: M/F")
    EST_CIVIL: Optional[str] = Field(None, description="Estado civil")
    ANTIGUEDAD_MES: Optional[float] = Field(None, description="Antigüedad como cliente, en meses")
    REGION: Optional[str] = Field(None, description="Región geográfica del cliente")
    FLAG_LIMA_PROVINCIA: int = Field(..., description="1 si es Lima Provincia, 0 si no")
    SUELDO_ESTIMADO: Optional[float] = Field(None, description="Sueldo estimado del cliente")
    CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD: Optional[float] = Field(
        None, description="Ratio de cobertura de deuda de consumo"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "MES": 201506, "CLIENTE": 12345,
                "LINEA_RENOVADO": 3770.0, "PLAZO_RENOVADO": 12,
                "USO_LINEA_TOTAL_TC_T2": 0.43, "USO_TRIM_LINEA_BBVA": 0.22,
                "NR_ENTIDADES_TOTAL_T2": 3, "DIFF_NRO_ENTIDA_TOTALES_T2_T12": 0,
                "SDO_CONSUMO_T2": 5000.0, "RESENCIA_OFERTA_PLD_RENOVADO": 6,
                "Ahorro_Sldo_Bco_T1": 2000.0, "PConsumo_Sldo_Bco_T1": 4500.0,
                "SDO_BCO_tot_sm_pasivo_Bco_6M": 3000.0, "EDAD": 34,
                "SEXO": "M", "EST_CIVIL": "C", "ANTIGUEDAD_MES": 40,
                "REGION": "CENTRO", "FLAG_LIMA_PROVINCIA": 0,
                "SUELDO_ESTIMADO": 3200.0, "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 1.2,
            }
        }
    }


class PrediccionOutput(BaseModel):
    """Respuesta del clasificador de renovación de préstamo."""
    score_riesgo: float
    decision: Literal["RENUEVA", "NO RENUEVA"]
    probabilidad_renovacion: float
    umbral_usado: float
    modelo: str


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status: str
    modelo: str
    version: str
    recall: float
    env: str
