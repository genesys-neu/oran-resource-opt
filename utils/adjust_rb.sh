#!/bin/bash


# Define the content for each tenant
slice_0='1110000000000000000000000'
slice_1='0001111110000000000000000'
slice_2='0000000001111111110000000'

# Write the content to the appropriate files in parallel
echo "$slice_0" > /root/radio_code/scope_config/slicing/slice_allocation_mask_tenant_0.txt &
echo "$slice_1" > /root/radio_code/scope_config/slicing/slice_allocation_mask_tenant_1.txt &
echo "$slice_2" > /root/radio_code/scope_config/slicing/slice_allocation_mask_tenant_2.txt &

# Wait for all background processes to finish
wait

echo "RB allocation updated"
