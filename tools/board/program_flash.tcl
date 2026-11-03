# Invoked by program_flash.py. Vivado Hardware Manager cfgmem flow.
open_hw_manager
connect_hw_server
open_hw_target
set hw_device [lindex [get_hw_devices] 0]
current_hw_device $hw_device
refresh_hw_device -update_hw_probes false $hw_device
set cfgpart [lindex [get_cfgmem_parts s25fl128sxxxxxx0-spi-x1_x2_x4] 0]
if {$cfgpart eq ""} { error "cfgmem part not found: s25fl128sxxxxxx0-spi-x1_x2_x4" }
create_hw_cfgmem -hw_device $hw_device -mem_dev $cfgpart
set cfgmem [lindex [get_hw_cfgmems -of_objects $hw_device] 0]
set_property PROGRAM.FILES [list $::mcs_file] $cfgmem
set_property PROGRAM.ADDRESS_RANGE {use_file} $cfgmem
set_property PROGRAM.PRM_FILE {} $cfgmem
program_hw_cfgmem -hw_cfgmem $cfgmem
verify_hw_cfgmem -hw_cfgmem $cfgmem
close_hw_target
disconnect_hw_server
close_hw_manager
