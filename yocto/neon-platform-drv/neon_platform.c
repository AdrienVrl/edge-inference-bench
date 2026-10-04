#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>

static int neon_platform_probe(struct platform_device *pdev) {
  dev_info(&pdev->dev, "neon-platform: probed, matched via device tree\n");
  return 0;
}

static int neon_platform_remove(struct platform_device *pdev) {
  dev_info(&pdev->dev, "neon-platform: removed\n");
  return 0;
}

// This is the actual matching mechanism: the kernel walks the device tree,
// and for every node it finds, compares that node's "compatible" string
// against every driver's of_match_table. A match calls this driver's probe().
static const struct of_device_id neon_platform_of_match[] = {
    {.compatible = "bench,neon-platform"}, {/* sentinel */}};
MODULE_DEVICE_TABLE(of, neon_platform_of_match);

static struct platform_driver neon_platform_driver = {
    .probe = neon_platform_probe,
    .remove = neon_platform_remove,
    .driver =
        {
            .name = "neon-platform",
            .of_match_table = neon_platform_of_match,
        },
};
module_platform_driver(neon_platform_driver);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Stub platform driver - binds via DT compatible string");
