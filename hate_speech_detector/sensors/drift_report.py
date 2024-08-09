from mage_ai.orchestration.run_status_checker import check_status

if "sensor" not in globals():
    from mage_ai.data_preparation.decorators import sensor


@sensor
def check_condition(*args, **kwargs) -> bool:
    """
    Template code for checking if block or pipeline run completed.
    """
    return check_status(
        "pipeline_uuid",
        kwargs["execution_date"],
        block_uuid="block_uuid",  # opt if u want sensor 2 wait a specif block
        hours=24,  # opt if u want to check 4 specif time window. Default 24hrs.
    )
